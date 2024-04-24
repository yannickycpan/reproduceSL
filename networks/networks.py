import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from utils.tfoperations import act_func_dict

def update_target_nn_move(tar_tvars, tvars, tau):
    target_params_update = [tf.assign_add(tar_tvars[idx], tau * (tvars[idx] - tar_tvars[idx]))
                                 for idx in range(len(tvars))]
    return target_params_update

def update_target_nn_assign(tar_tvars, tvars):
    target_params_update = [tf.assign(tar_tvars[idx],  tvars[idx]) for idx in range(len(tvars))]
    return target_params_update

def create_implicit_concatinput_hidden(xinput, target_input, actfunc, n_hidden1, n_hidden2):
    totalinput = tf.concat([xinput, target_input], axis=1)
    hidden1 = tf.keras.layers.Dense(n_hidden1, activation=actfunc)(totalinput)
    # hidden1 = tf.contrib.layers.fully_connected(hidden1, n_hidden1, activation_fn=actfunc)
    hidden2 = tf.keras.layers.Dense(n_hidden2, activation=actfunc)(hidden1)
    return hidden2


''' this version of taking derivative w.r.t. embedding does not work well, 
there should be another loss for the embedding '''


def create_implicit_concatinput_hidden_test(xinput, target_input, actfunc, n_hidden1, n_hidden2, g):
    xhidden1 = tf.contrib.layers.fully_connected(xinput, n_hidden1, activation_fn=actfunc)
    yhidden1 = tf.contrib.layers.fully_connected(target_input, n_hidden1, activation_fn=actfunc)
    g.watch(yhidden1)
    totalinput = tf.concat([xhidden1, yhidden1], axis=1)
    hidden1 = tf.contrib.layers.fully_connected(totalinput, n_hidden1, activation_fn=actfunc)
    hidden2 = tf.contrib.layers.fully_connected(hidden1, n_hidden2, activation_fn=actfunc)
    return hidden2 * hidden2, yhidden1


def debug(x, input_d, output_d):
    W = tf.get_variable('W2', shape=(input_d, output_d))
    b = tf.get_variable('W1', shape=(output_d))
    return tf.nn.relu(tf.matmul(x, W) + b), W


def create_implicit_dotprod_hidden(xinput, target_input, actfunc, n_hidden1, n_hidden2):
    ''' in this architecture, the act func can be relu '''
    W = None
    xhidden1 = tf.contrib.layers.fully_connected(xinput, n_hidden1, activation_fn=actfunc)
    xhidden2 = tf.contrib.layers.fully_connected(xhidden1, n_hidden2, activation_fn=actfunc)

    ''' if change the below line to use relu, it does much worse; it seems the y path must be tanh '''
    yhidden1 = tf.contrib.layers.fully_connected(target_input, n_hidden1, activation_fn=actfunc)
    yhidden2 = tf.contrib.layers.fully_connected(yhidden1, n_hidden2, activation_fn=actfunc)

    hidden2 = yhidden2 * xhidden2
    return hidden2, xhidden1, yhidden1


# w_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003), the init is not good here
def create_implicit_nn(scopename, config):
    n_input, n_hidden1, n_hidden2 = config.dim, config.n_h1, config.n_h2
    actfunc = config.actfunctype
    with tf.variable_scope(scopename):
        input = tf.placeholder(tf.float32, [None, n_input])
        with tf.GradientTape() as g:
            target_input = tf.placeholder(tf.float32, [None, 1])
            g.watch(target_input)
            xreps = None
            W = None
            if 'Dot' in scopename:
                hidden2, xreps, yreps = create_implicit_dotprod_hidden(input, target_input,
                                                                       act_func_dict[actfunc], n_hidden1, n_hidden2)
            else:
                hidden2 = create_implicit_concatinput_hidden(input, target_input,
                                                             act_func_dict[actfunc], n_hidden1, n_hidden2)
            fxy = tf.keras.layers.Dense(1, activation=None)(hidden2)
            gradwrt_target = tf.gradients(fxy, target_input)[0]

        firstloss = tf.squeeze(tf.square(fxy))
        batch_jacobian = g.batch_jacobian(fxy, target_input)
        batch_jacobian = tf.reshape(batch_jacobian, [-1])
        derivativeloss = tf.squeeze(tf.square(batch_jacobian + 1.0))
        loss = tf.reduce_mean(firstloss) + tf.reduce_mean(derivativeloss)
        lossvec = tf.squeeze(firstloss) + derivativeloss

        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopename)
        gradplaceholders = [tf.placeholder(tf.float32, tvar.get_shape().as_list()) for tvar in tvars]
        if config.highorder_reg > 0:
            derivative_2ndorder = tf.squeeze(tf.linalg.diag_part(tf.gradients(gradwrt_target, target_input)))
            loss = loss + tf.reduce_mean(tf.square(derivative_2ndorder)) * config.highorder_reg
    return input, target_input, loss, lossvec, fxy, hidden2, xreps, tf.reduce_mean(derivativeloss), W, gradplaceholders, tvars


def create_lta_regression_nn(scopename, config):
    n_input, n_hidden1, n_hidden2, actfunctype \
        = config.dim, config.n_h1, config.n_h2, config.actfunctype
    with tf.variable_scope(scopename):
        input = tf.placeholder(tf.float32, [None, n_input])
        target_input = tf.placeholder(tf.float32, [None])
        # Dense Layer
        dense1 = tf.contrib.layers.fully_connected(input, n_hidden1, activation_fn=act_func_dict[actfunctype])
        # config.LTA.set_extra_act_strength(dense1, n_hidden2)
        phi = tf.contrib.layers.fully_connected(dense1, n_hidden2, activation_fn=config.LTA.func)

        output = tf.contrib.layers.fully_connected(phi, 1, activation_fn=None)
        loss = tf.losses.mean_squared_error(target_input, tf.squeeze(output)) # + config.LTA.lta_loss
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopename)
    return input, target_input, output, loss, phi, tvars


def create_softmax_classification_nn(scopename, config):
    n_input, n_classes, n_hidden1, n_hidden2, actfunc \
        = config.dim, config.n_classes, config.n_h1, config.n_h2, config.actfunctype

    with tf.variable_scope(scopename):
        if isinstance(n_input, list):
            input = tf.placeholder(tf.float32, [None] + n_input)
            inputx = tf.keras.layers.Flatten()(input)
        else:
            input = tf.placeholder(tf.float32, [None, n_input])
            inputx = input
        target_input = tf.placeholder(tf.int32, [None, n_classes])

        hidden1 = tf.keras.layers.Dense(n_hidden1, activation=act_func_dict[actfunc])(inputx)
        hidden2 = tf.keras.layers.Dense(n_hidden2, activation=act_func_dict[actfunc])(hidden1)
        logits = tf.keras.layers.Dense(n_classes, activation=None)(hidden2)

        loss = tf.losses.softmax_cross_entropy(onehot_labels=target_input, logits=logits)
        predictions = tf.argmax(input=logits, axis=1)
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopename)
    return input, target_input, logits, predictions, loss, None, tvars


def create_regression_nn(scopename, config):
    n_input, n_hidden1, n_hidden2, huber_delta, actfunc, outputactfunc \
        = config.dim, config.n_h1, config.n_h2, config.huber_delta, config.actfunctype, config.outputactfunctype
    with tf.variable_scope(scopename):
        input = tf.placeholder(tf.float32, [None, n_input])
        target_input = tf.placeholder(tf.float32, [None])

        hidden1 = tf.keras.layers.Dense(n_hidden1, activation=act_func_dict[actfunc])(input)
        hidden2 = tf.keras.layers.Dense(n_hidden2, activation=act_func_dict[actfunc])(hidden1)
        output = tf.keras.layers.Dense(1, activation=None)(hidden2)

        if 'HuberRegression' in scopename:
            loss = tf.losses.huber_loss(target_input, tf.squeeze(output), delta=huber_delta)
        else:
            loss = tf.losses.mean_squared_error(target_input, tf.squeeze(output))
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopename)
        unfoldedloss = tf.abs(target_input - tf.squeeze(output))
    return input, target_input, output, output, loss, unfoldedloss, tvars


def create_poisson_nn(scopename, config):
    n_input, n_hidden1, n_hidden2, actfunc = config.dim, config.n_h1, config.n_h2, config.actfunctype
    with tf.variable_scope(scopename):
        input = tf.placeholder(tf.float32, [None, n_input])
        target_input = tf.placeholder(tf.float32, [None])
        hidden1 = tf.keras.layers.Dense(n_hidden1, activation=act_func_dict[actfunc])(input)
        hidden2 = tf.keras.layers.Dense(n_hidden2, activation=act_func_dict[actfunc])(hidden1)
        output = tf.keras.layers.Dense(1, activation=tf.exp)(hidden2)
        loss = -tf.reduce_mean(target_input*tf.squeeze(tf.log(output)) - output)
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopename)
        unfoldedloss = tf.abs(target_input - tf.squeeze(output))
    return input, target_input, tf.log(output), output, loss, unfoldedloss, tvars


''' in below function, n_k is the number of mixture components in the network, c
    urrently only suitable for scalar y, i.e. n_output = 1 '''


def create_mdn_nn(scopename, config, n_output=1):
    n_input, n_hidden1, n_hidden2, n_k, actfunc \
        = config.dim, config.n_h1, config.n_h2, config.n_mdncomponents, config.actfunctype
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    with tf.variable_scope(scopename):
        input = tf.placeholder(tf.float32, [None, n_input])
        target_input = tf.placeholder(tf.float32, [None])
        hidden1 = tf.contrib.layers.fully_connected(input, n_hidden1, activation_fn=act_func_dict[actfunc])
        hidden2 = tf.contrib.layers.fully_connected(hidden1, n_hidden2, activation_fn=act_func_dict[actfunc])

        alphas = tf.contrib.layers.fully_connected(hidden2, n_k, activation_fn=tf.nn.softmax)
        mu = tf.contrib.layers.fully_connected(hidden2, n_k * n_output, activation_fn=None)
        sigma = tf.contrib.layers.fully_connected(hidden2, n_k * n_output, activation_fn=tf.exp)

        ''' manually build MDN: does not work  '''
        #prod_density = tf.reduce_sum(alphas * tf.exp(-0.5 * tf.square(target_input - mu) / tf.square(sigma)), axis=1)
        #log_loss = -tf.reduce_sum(tf.log(prod_density))
        gm = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(
                probs=alphas),
            components_distribution=tfd.Normal(
                loc=mu,  # One for each component.
                scale=sigma))  # And same here
        neg_log_prob = -gm.log_prob(target_input)
        log_loss = tf.reduce_sum(neg_log_prob)

        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopename)
    return input, target_input, alphas, mu, log_loss, neg_log_prob, tvars


def create_qunatile_regression_loss(target_input, quantile_values, tau_hat_mat, n_quantiles,  kaap=0.01):
    """  u_mat is a b * N size matrix  """
    u_mat = quantile_values - target_input
    u_mat = -u_mat
    huberloss_u_mat = 0.5 * tf.square(u_mat) * tf.cast((tf.abs(u_mat) <= kaap), tf.float32) \
                      + kaap * (tf.abs(u_mat) - 0.5 * kaap) * tf.cast((tf.abs(u_mat) > kaap), tf.float32)
    dirac_umat = tf.abs(tau_hat_mat - tf.cast((u_mat < 0.0), tf.float32))
    final_loss_mat = dirac_umat * huberloss_u_mat
    final_loss = 1.0 / n_quantiles * tf.reduce_sum(final_loss_mat, axis=1)
    final_loss = tf.reduce_mean(final_loss)
    return final_loss


def create_qunatile_regression_loss_vanilla(target_input, quantile_values, tau_hat_mat, n_quantiles,  kaap=0.01):
    """  u_mat is a b * N size matrix  """
    u_mat = target_input - quantile_values
    dirac_umat = tau_hat_mat - tf.cast((u_mat < 0.0), tf.float32)
    final_loss_mat = dirac_umat * u_mat
    final_loss = tf.reduce_mean(final_loss_mat, axis=1)
    final_loss = tf.reduce_mean(final_loss)
    return final_loss


def create_quantile_nn(scopename, n_input, n_quantiles, tau_hat_mat, n_hidden1=16, n_hidden2=16):
    with tf.variable_scope(scopename):
        input = tf.placeholder(tf.float32, [None, n_input])
        hidden1 = tf.contrib.layers.fully_connected(input, n_hidden1, activation_fn=tf.nn.tanh)
        hidden2 = tf.contrib.layers.fully_connected(hidden1, n_hidden2, activation_fn=tf.nn.tanh)
        quantile_values = tf.contrib.layers.fully_connected(hidden2, n_quantiles, activation_fn=None)
        expected_values = tf.reduce_mean(quantile_values, axis=1)
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopename)

        target_input = tf.placeholder(tf.float32, [None, 1], name='target_holders')
        #loss = create_qunatile_regression_loss(target_input, quantile_values, tau_hat_mat, n_quantiles)
        loss = create_qunatile_regression_loss_vanilla(target_input, quantile_values, tau_hat_mat, n_quantiles)

    return input, target_input, quantile_values, expected_values, loss, tvars

import numpy as np
def create_linear_regression_nn(scopename, config):
    outdim = 1 if config.n_classes is None else config.n_classes
    with tf.variable_scope(scopename):
        input = tf.placeholder(tf.float32, [None]+config.dim) if isinstance(config.dim, list) \
            else tf.placeholder(tf.float32, [None, config.dim])
        inputx = tf.reshape(input, [-1, int(np.prod(config.dim))])
        target_input = tf.squeeze(tf.placeholder(tf.float32, [None, outdim]))
        #w_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003), kernel_initializer='zero'
        # weights_regularizer = tf.contrib.layers.l2_regularizer(reg)
        output = tf.keras.layers.Dense(outdim, activation=None)(inputx)
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopename)
        loss = tf.losses.mean_squared_error(target_input, tf.squeeze(output))
        unfoldedloss = tf.abs(target_input - tf.squeeze(output))
    return input, target_input, output, output, loss, unfoldedloss, tvars


def create_poisson_regression_linear(scopename, config):
    n_input = config.dim
    with tf.variable_scope(scopename):
        input = tf.placeholder(tf.float32, [None, n_input])
        target_input = tf.placeholder(tf.float32, [None])
        #w_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003), kernel_initializer='zero'
        output = tf.keras.layers.Dense(1, activation=tf.exp)(input)
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopename)
        loss = -tf.reduce_mean(target_input*tf.squeeze(tf.log(output)) - output)
        unfoldedloss = -target_input * tf.squeeze(tf.log(output)) + output
    return input, target_input, tf.log(output), output, loss, unfoldedloss, tvars


'''cross entropy loss'''
def create_classification_nn(scopename, n_features, n_classes, n_hidden1=16, n_hidden2=16, usetanh = True):
    with tf.variable_scope(scopename):
        # Input Layer
        input_layer = tf.placeholder(tf.float32, [None, n_features])

        hiddenact = tf.nn.relu if not usetanh else tf.nn.tanh
        dense1 = tf.contrib.layers.fully_connected(input_layer, n_hidden1, activation_fn=hiddenact)
        dense2 = tf.contrib.layers.fully_connected(dense1, n_hidden2, activation_fn=hiddenact)

        # Logits layer
        logits = tf.contrib.layers.fully_connected(dense2, n_classes, activation_fn=None)
        predictions = tf.argmax(input=logits, axis=1)

        # Calculate Loss (for both TRAIN and EVAL modes)
        target_input = tf.placeholder(tf.int32, [None, n_classes])
        loss = tf.losses.softmax_cross_entropy(onehot_labels=target_input, logits=logits)
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopename)
    return input_layer, target_input, predictions, loss, tvars