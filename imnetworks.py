import tensorflow as tf
from tfoperations import LTA_multitilings, act_func_dict
import numpy as np

def lenet5_convlayers(input_layer, actfn = tf.nn.relu):
    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 32, 32, 1]
    # Output Tensor Shape: [batch_size, 28, 28, 6]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=6,
        kernel_size=[5, 5],
        padding="valid",
        activation=actfn)
    # output is 14, 14, 6, valid padding by default
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Input Tensor Shape: [batch_size, 14, 14, 6]
    # Output Tensor Shape: [batch_size, 10, 10, 16]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=16,
        kernel_size=[5, 5],
        padding="valid",
        activation=actfn)
    # Pooling Layer #2, output becomes [5, 5, 16]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    #pool2_flat = tf.reshape(pool2, [-1, 5 * 5 * 16])
    pool2_flat = tf.layers.flatten(pool2)
    return pool2_flat

def lenet4_convlayers(input_layer, actfn = tf.nn.relu):
    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 32, 32, 1]
    # Output Tensor Shape: [batch_size, 28, 28, 6]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=4,
        kernel_size=[5, 5],
        padding="valid",
        activation=actfn)
    # output is 14, 14, 4, valid padding by default
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Input Tensor Shape: [batch_size, 14, 14, 4]
    # Output Tensor Shape: [batch_size, 10, 10, 16]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=16,
        kernel_size=[5, 5],
        padding="valid",
        activation=actfn)
    # Pooling Layer #2, output becomes [5, 5, 16]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    # pool2_flat = tf.reshape(pool2, [-1, 5 * 5 * 16])
    pool2_flat = tf.layers.flatten(pool2)
    return pool2_flat


def create_image_classification_nn(scopename, config):
    inputshape, n_classes, n_hidden1, n_hidden2, actfunctype \
        = config.dim, config.n_classes, config.n_h1, config.n_h2, config.actfunctype
    with tf.variable_scope(scopename):
        """Model function for CNN."""
        # Input Layer
        input_layer = tf.placeholder(tf.float32, [None] + inputshape)
        pool2_flat = lenet5_convlayers(input_layer)

        # Dense Layer
        dense1 = tf.contrib.layers.fully_connected(pool2_flat, n_hidden1,
                                                   activation_fn=act_func_dict[actfunctype])
        dense2 = tf.contrib.layers.fully_connected(dense1, n_hidden2,
                                                   activation_fn=act_func_dict[actfunctype])
        # Logits layer
        logits = tf.contrib.layers.fully_connected(dense2, n_classes, activation_fn=None)
        predictions = tf.argmax(input=logits, axis=1)

        # Calculate Loss (for both TRAIN and EVAL modes)
        target_input = tf.placeholder(tf.int32, [None, n_classes])
        loss = tf.losses.softmax_cross_entropy(onehot_labels=target_input, logits=logits)
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopename)
    return input_layer, target_input, predictions, loss, tvars


def create_image_classification_resnn(scopename, config):
    inputshape, n_classes, n_hidden1, n_hidden2, actfunctype \
        = config.dim, config.n_classes, config.n_h1, config.n_h2, config.actfunctype
    with tf.variable_scope(scopename):
        """Model function for CNN."""
        # Input Layer
        input_layer = tf.placeholder(tf.float32, [None] + inputshape)
        pool2_flat = lenet5_convlayers(input_layer)

        # Dense Layer
        dense1 = tf.contrib.layers.fully_connected(pool2_flat, n_hidden1,
                                                   activation_fn=act_func_dict[actfunctype])
        dense2 = tf.contrib.layers.fully_connected(dense1, n_hidden2,
                                                   activation_fn=act_func_dict[actfunctype])
        dense3 = tf.contrib.layers.fully_connected(dense2 + dense1, n_hidden2,
                                                   activation_fn=act_func_dict[actfunctype])

        # Logits layer
        logits = tf.contrib.layers.fully_connected(dense3 + dense2 + dense1, n_classes, activation_fn=None)
        predictions = tf.argmax(input=logits, axis=1)

        # Calculate Loss (for both TRAIN and EVAL modes)
        target_input = tf.placeholder(tf.int32, [None, n_classes])
        loss = tf.losses.softmax_cross_entropy(onehot_labels=target_input, logits=logits)
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopename)
    return input_layer, target_input, predictions, loss, tvars


def create_image_classification_ltann(scopename, config):
    inputshape, n_classes, n_hidden1, n_hidden2, actfunctype \
        = config.dim, config.n_classes, config.n_h1, config.n_h2, config.actfunctype
    with tf.variable_scope(scopename):
        # Input Layer
        input_layer = tf.placeholder(tf.float32, [None] + inputshape)
        pool2_flat = lenet5_convlayers(input_layer)

        # Dense Layer
        dense1 = tf.contrib.layers.fully_connected(pool2_flat, n_hidden1, activation_fn=act_func_dict[actfunctype])
        config.LTA.set_extra_act_strength(dense1, n_hidden2)
        phi = tf.contrib.layers.fully_connected(dense1, n_hidden2, activation_fn=config.LTA.func)

        # Logits layer
        logits = tf.contrib.layers.fully_connected(phi, n_classes, activation_fn=None)
        predictions = tf.argmax(input=logits, axis=1)

        # Calculate Loss (for both TRAIN and EVAL modes)
        target_input = tf.placeholder(tf.int32, [None, n_classes])
        loss = tf.losses.softmax_cross_entropy(onehot_labels=target_input, logits=logits) \
               + config.LTA.lta_loss
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopename)
    return input_layer, target_input, predictions, loss, phi, tvars

def create_image_classification_multiltann(scopename, config):
    inputshape, n_classes, n_hidden1, n_hidden2, actfunctype \
        = config.dim, config.n_classes, config.n_h1, config.n_h2, config.actfunctype
    with tf.variable_scope(scopename):
        # Input Layer
        input_layer = tf.placeholder(tf.float32, [None] + inputshape)
        pool2_flat = lenet5_convlayers(input_layer)

        # Dense Layer
        dense1 = tf.contrib.layers.fully_connected(pool2_flat, n_hidden1, activation_fn=act_func_dict[actfunctype])
        ltainput = tf.contrib.layers.fully_connected(dense1, n_hidden2, act_func_dict[config.actfunctypeLTA])

        # fine lta
        phi_input = tf.placeholder(tf.float32, [None, n_hidden2])
        phi_fine = config.LTA_fine.LTA_binning_func(phi_input)
        print('phi fine dim is ::=====================  ', phi_fine)
        logits = tf.contrib.layers.fully_connected(phi_fine, n_classes, activation_fn=None)
        predictions = tf.argmax(input=logits, axis=1)

        # coarse lta
        phi_coarse = config.LTA_coarse.LTA_binning_func(ltainput, coarse_n_tiles=config.coarse_n_tiles,
                                                                  coarse_eta=config.coarse_eta)
        print('phi coarse dim is ::===================  ', phi_coarse)
        logits_coarse = tf.contrib.layers.fully_connected(phi_coarse, n_classes, activation_fn=None)

        # Calculate Loss (for both TRAIN and EVAL modes)
        target_input = tf.placeholder(tf.int32, [None, n_classes])
        coarse_loss = tf.losses.softmax_cross_entropy(onehot_labels=target_input, logits=logits_coarse)\
                      + config.LTA_coarse.lta_loss
        fine_loss = tf.losses.softmax_cross_entropy(onehot_labels=target_input, logits=logits) \
               + config.LTA_fine.lta_loss

        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopename)
    return input_layer, target_input, phi_input, ltainput, predictions, coarse_loss, fine_loss, phi_fine, phi_coarse, tvars


def create_image_classification_coarselta_auxi_nn(scopename, config):
    inputshape, n_classes, n_hidden1, n_hidden2, actfunctype \
        = config.dim, config.n_classes, config.n_h1, config.n_h2, config.actfunctype
    with tf.variable_scope(scopename):
        # Input Layer
        input_layer = tf.placeholder(tf.float32, [None] + inputshape)
        pool2_flat = lenet5_convlayers(input_layer)

        # Dense Layer
        dense1 = tf.contrib.layers.fully_connected(pool2_flat, n_hidden1, activation_fn=act_func_dict[actfunctype])
        ltainput = tf.contrib.layers.fully_connected(dense1, n_hidden2, act_func_dict[config.actfunctypeLTA])

        # fine lta
        phi_fine = config.LTA_fine.LTA_binning_func(ltainput)
        logits = tf.contrib.layers.fully_connected(phi_fine, n_classes, activation_fn=None)
        predictions = tf.argmax(input=logits, axis=1)

        # coarse lta
        phi_coarse = config.LTA_coarse.LTA_binning_func(ltainput, coarse_n_tiles=config.coarse_n_tiles,
                                                                  coarse_eta=config.coarse_eta)
        logits_coarse = tf.contrib.layers.fully_connected(phi_coarse, n_classes, activation_fn=None)

        # Calculate Loss (for both TRAIN and EVAL modes)
        target_input = tf.placeholder(tf.int32, [None, n_classes])
        coarse_loss = tf.losses.softmax_cross_entropy(onehot_labels=target_input, logits=logits_coarse)\
                      + config.LTA_coarse.lta_loss
        fine_loss = tf.losses.softmax_cross_entropy(onehot_labels=target_input, logits=logits) \
               + config.LTA_fine.lta_loss

        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopename)
    return input_layer, target_input, predictions, coarse_loss, fine_loss, tvars


def create_image_classification_ltamiddlenn(scopename, config):
    inputshape, n_classes, n_hidden1, n_hidden2, actfunctype \
        = config.dim, config.n_classes, config.n_h1, config.n_h2, config.actfunctype
    useStrength, actfunctypeLTA = config.use_strength, config.actfunctypeLTA
    # lta_eta, reg_factor = config.lta_eta, config.outofbound_reg
    # lta_input_min, lta_input_max = config.lta_input_min, config.lta_input_max
    # n_tiles = config.n_tiles
    with tf.variable_scope(scopename):
        # Input Layer
        input_layer = tf.placeholder(tf.float32, [None] + inputshape)
        pool2_flat = lenet5_convlayers(input_layer)
        # sparse Layer
        phi = tf.contrib.layers.fully_connected(pool2_flat, n_hidden1, activation_fn=config.LTA.func)
        dense2 = tf.contrib.layers.fully_connected(phi, n_hidden2, activation_fn=act_func_dict[actfunctype])
        # Logits layer
        logits = tf.contrib.layers.fully_connected(dense2, n_classes, activation_fn=None)
        predictions = tf.argmax(input=logits, axis=1)

        # Calculate Loss (for both TRAIN and EVAL modes)
        target_input = tf.placeholder(tf.int32, [None, n_classes])
        loss = tf.losses.softmax_cross_entropy(onehot_labels=target_input, logits=logits) \
               + config.LTA.lta_loss
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopename)
    return input_layer, target_input, predictions, loss, phi, tvars


def create_image_levelset_classification_nn(scopename, inputshape, n_classes):
    with tf.variable_scope(scopename):
        #print('the input shape is ', inputshape)
        input_layer = tf.placeholder(tf.float32, [None] + inputshape)
        target_input = tf.placeholder(tf.float32, [None, n_classes])
        pool2_flat = lenet5_convlayers(input_layer, tf.nn.tanh)
        #pool2_flat = tf_doc_convlayers(input_layer)

        # Dense Layer
        concatflat = tf.concat([pool2_flat, target_input], axis=1)
        dense1 = tf.contrib.layers.fully_connected(concatflat, 120, activation_fn=tf.nn.tanh)
        #yhidden1 = tf.contrib.layers.fully_connected(target_input, 120, activation_fn=tf.nn.tanh)
        #dense1 = yhidden1*dense1
        dense2 = tf.contrib.layers.fully_connected(dense1, 84, activation_fn=tf.nn.tanh)

        output = tf.contrib.layers.fully_connected(dense2, n_classes, activation_fn=tf.nn.tanh)
        '''this one can work only for single sample processing, incorrect for mini-batch update, 
        it seems the only way is to manually aggregate gradient for mini-batch update'''
        jacobian = tf.stack([tf.reduce_mean(tf.gradients(output[:, idx], target_input)[0], axis=0)
                            for idx in range(n_classes)], axis=0, name='jacobian_vars')

        #regjacobian = tf.stack([tf.reduce_mean(tf.contrib.layers.flatten(tf.gradients(output[:, idx], input_layer)[0]), axis=0)
        #                     for idx in range(n_classes)], axis=0, name='regjacobian_vars')

        firstloss = tf.reduce_mean(tf.square(output), axis=1)
        #firstloss = tf.reduce_mean(output, axis=1)

        #regloss = tf.reduce_sum(tf.square(regjacobian))

        loss = tf.reduce_mean(firstloss) + \
               tf.reduce_mean(tf.square(jacobian + tf.eye(n_classes)))

        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopename)
        gradplaceholders = [tf.placeholder(tf.float32, tvar.get_shape().as_list()) for tvar in tvars]
    return input_layer, target_input, firstloss, loss, gradplaceholders, tvars


def cnn_encoder(X, n_hidden1, latent_dim, activation=tf.nn.relu):
    x = tf.layers.conv2d(X, filters=64, kernel_size=4, strides=2, padding='valid', activation=activation)
    x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='valid', activation=activation)
    x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='valid', activation=activation)
    x = tf.layers.flatten(x)

    print('the flattened vec size in encoder is :: ', x.shape)
    # Local latent variables
    x = tf.layers.dense(x, units=n_hidden1, activation=activation)
    mean_ = tf.layers.dense(x, units=latent_dim, name='mean', activation=None)
    #std_dev = tf.nn.softplus(tf.layers.dense(x, units=latent_dim), name='std_dev')  # softplus to force >0
    # Reparametrization trick
    #epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], latent_dim]), name='epsilon')
    #z = mean_ + tf.multiply(epsilon, std_dev)

    #strength = tf.layers.dense(x, units=latent_dim, name='strength', activation=tf.nn.tanh)
    strength = tf.ones_like(mean_)

    return mean_, strength


def fc_encoder(name, X, n_hidden1, latent_dim, useStrength=0, latent_activation=0, sparse_dim=0):
    activation = tf.nn.relu
    inputx = tf.layers.flatten(X)
    print('the flattened vec size in encoder is :: ', inputx.shape)
    # Local latent variables
    x = tf.layers.dense(inputx, units=n_hidden1, activation=activation)
    x = tf.layers.dense(x, units=n_hidden1, activation=activation)

    if name == 'AE-Linear':
        mean_ = tf.layers.dense(x, units=sparse_dim, name='mean', activation=act_func_dict[latent_activation])
    else:
        mean_ = tf.layers.dense(x, units=latent_dim, name='mean', activation=act_func_dict[latent_activation])
    print('latent activation function is :: =================================== ',
          latent_activation, act_func_dict[latent_activation])

    strength = tf.layers.dense(x, units=latent_dim, name='strength', activation=tf.nn.sigmoid) \
        if useStrength==1 else tf.ones_like(mean_)

    #scalor = tf.ones_like(mean_)
    scalor = 1.

    return mean_, strength, scalor


def cnn_decoder(z, imwidth):
    activation = tf.nn.relu
    x = tf.layers.dense(z, units=64, activation=activation)
    x = tf.layers.dense(x, units=64, activation=activation)
    recovered_size = int(np.sqrt(64))
    x = tf.reshape(x, [-1, recovered_size, recovered_size, 1])

    x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
    x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
    x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)

    x = tf.contrib.layers.flatten(x)
    x = tf.layers.dense(x, units=imwidth * imwidth, activation=None)
    x = tf.layers.dense(x, units=imwidth * imwidth, activation=tf.nn.sigmoid)
    img = tf.reshape(x, shape=[-1, imwidth, imwidth, 1])

    return img


def linear_decoder(z, imwidth):
    x = tf.layers.dense(z, units=imwidth * imwidth, activation=tf.nn.sigmoid)
    img = tf.reshape(x, shape=[-1, imwidth, imwidth, 1])
    return img

def nonlinear_decoder(z, imwidth):
    x = tf.layers.dense(z, units=128, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=imwidth * imwidth, activation=tf.nn.sigmoid)
    img = tf.reshape(x, shape=[-1, imwidth, imwidth, 1])
    return img


def onelayer_randproj_decoder(z, imwidth, n_hidden2, sparse_dim):
    print('nh and sparsedim are :: ', n_hidden2, sparse_dim)
    randmat = np.random.normal(0.0, np.sqrt(1.0/sparse_dim), (n_hidden2, sparse_dim)).astype(np.float32)
    rdproj = tf.constant(randmat)
    phi = tf.matmul(z, rdproj)
    x = tf.layers.dense(phi, units=imwidth * imwidth, activation=tf.nn.sigmoid)
    img = tf.reshape(x, shape=[-1, imwidth, imwidth, 1])
    return img, phi


def onelayer_nonlinear_decoder(z, imwidth, sparse_dim, reg=0.0):
    phi = tf.layers.dense(z, units=sparse_dim, activation=tf.nn.relu,
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg))
    print('one more hidden layer used !!!!!!!! ')
    x = tf.layers.dense(phi, units=imwidth * imwidth, activation=tf.nn.sigmoid)
    img = tf.reshape(x, shape=[-1, imwidth, imwidth, 1])
    return img, phi


def create_decor_loss(phi):
    phiTphi = tf.matmul(phi, phi, transpose_a=True)
    decor_loss = phiTphi - tf.matrix_diag_part(phiTphi)
    return tf.reduce_mean(tf.square(decor_loss))