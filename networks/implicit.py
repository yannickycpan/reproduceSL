import tensorflow as tf
from utils.tfoperations import act_func_dict, create_two_layer_fc
from networks.cnns import lenet5_convlayers, get_lenet5_w_b_cnn, conv_net_with_fc


def create_implicit_dot_nonimage(xinput, yinput, xinput_label, g, config):
    n_hidden1, n_hidden2, actfunctype \
        = config.n_h1, config.n_h2, config.actfunctype
    g.watch(yinput)
    input_y = tf.concat([yinput, tf.cast(xinput_label, tf.float32)], axis=1)

    xhidden1 = tf.contrib.layers.fully_connected(xinput, n_hidden1, activation_fn=act_func_dict[actfunctype])
    xphi = tf.contrib.layers.fully_connected(xhidden1, n_hidden2, activation_fn=act_func_dict[actfunctype])

    yhidden1 = tf.contrib.layers.fully_connected(input_y, n_hidden1, activation_fn=act_func_dict[actfunctype])
    yphi = tf.contrib.layers.fully_connected(yhidden1, n_hidden2, activation_fn=act_func_dict[actfunctype])
    jointphi = xphi * yphi
    return jointphi, xphi, yinput


def create_implicit_joint_nonimage(xinput, yinput, xinput_label, g, config, jointphi_actfunc=tf.nn.tanh):
    n_hidden1, n_hidden2, actfunctype \
        = config.n_h1, config.n_h2, config.actfunctype
    # casted_label = tf.cast(xinput_label, tf.float32)
    # der_wrt_to = yinput
    g.watch(yinput)
    concatedinput = tf.concat([xinput, yinput], axis=1)
    hidden1 = tf.contrib.layers.fully_connected(concatedinput, n_hidden1, jointphi_actfunc)
    hidden2 = tf.contrib.layers.fully_connected(hidden1, n_hidden2, activation_fn=jointphi_actfunc)
    # labelhidden1 = tf.contrib.layers.fully_connected(casted_label, n_hidden1, act_func_dict[actfunctype])
    # labelhidden2 = tf.contrib.layers.fully_connected(labelhidden1, n_hidden2, activation_fn=act_func_dict[actfunctype])
    jointphi = hidden2
    return jointphi, hidden2, yinput


def create_implicit_dotprod(xinput, yinput, xinput_label, config):
    inputshape, n_hidden1, n_hidden2 \
        = config.dim, config.n_h1, config.n_h2
    yaftercnn = lenet5_convlayers(yinput, tf.nn.tanh)
    yhidden1 = tf.contrib.layers.fully_connected(tf.concat([yaftercnn, tf.cast(xinput_label, tf.float32)], axis=1),
                                                 n_hidden1, activation_fn=tf.nn.tanh)
    yhidden2 = tf.contrib.layers.fully_connected(yhidden1, n_hidden2, activation_fn=tf.nn.tanh)

    xinput = lenet5_convlayers(xinput)
    print('the shape after covn is =================================', xinput.shape)
    xhidden1 = tf.contrib.layers.fully_connected(xinput, n_hidden1, activation_fn=tf.nn.relu)
    xhidden2 = tf.contrib.layers.fully_connected(xhidden1, n_hidden2, activation_fn=tf.nn.relu)

    # matchedxhidden2 = tf.ensure_shape(xhidden2, yhidden2.get_shape().as_list())
    hidden2 = yhidden2 * xhidden2
    return hidden2, yhidden2, xhidden2


def create_implicit_joint(xinput, yinput, n_hidden1, n_hidden2, actfunc):
    # tf.concat([xinput, yinput], -1)
    yaftercnn = lenet5_convlayers(yinput, tf.nn.tanh)
    xaftercnn = lenet5_convlayers(xinput)

    hidden1 = tf.contrib.layers.fully_connected(tf.concat([xaftercnn, yaftercnn], axis=1),
                                                n_hidden1, activation_fn=actfunc)
    hidden2 = tf.contrib.layers.fully_connected(hidden1, n_hidden2, activation_fn=actfunc)
    return hidden2, yaftercnn, xaftercnn