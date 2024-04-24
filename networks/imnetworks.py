import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from utils.tfoperations import act_func_dict
from tensorflow.keras import layers, models
from networks.cnns import lenet5_convlayers
import numpy as np


def pgd_attack(loss, x, epsilon=0.05, alpha=0.01, num_steps=5):
    # Apply the PGD attack
    x_adv = tf.identity(x)
    for i in range(num_steps):
        grad = tf.gradients(loss, x)[0]
        x_adv = x_adv + alpha * tf.sign(grad)
    x_adv = tf.clip_by_value(x_adv, x - epsilon, x + epsilon)
    return tf.clip_by_value(x_adv, 0, 1)

def get_mask(listvars, thres):
    masks = []
    for v in listvars:
        masks.append(tf.cast(tf.abs(v) > thres, tf.float32))
    return masks

def mask_vars(listvars, masks):
    ops = []
    for id, v in enumerate(listvars):
        ops.append(tf.assign(v, v*masks[id]))
    return ops

def vectorize_vars(listvars):
    vec = []
    for v in listvars:
        vec.append(tf.reshape(v, [-1]))
    return tf.concat(vec, axis=0)


def create_image_classification_nn(scopename, config):
    inputshape, n_classes, n_hidden1, n_hidden2, actfunctype \
        = config.dim, config.n_classes, config.n_h1, config.n_h2, config.actfunctype
    with tf.variable_scope(scopename):
        """Model function for CNN."""
        # Input Layer
        input_layer = tf.placeholder(tf.float32, [None] + inputshape)
        # pool2_flat = lenet5_convlayers(input_layer)
        pool2_flat = tf.layers.flatten(input_layer)
        # Dense Layer
        dense1 = tf.keras.layers.Dense(n_hidden1, activation=act_func_dict[actfunctype])(pool2_flat)
        dense2 = tf.keras.layers.Dense(n_hidden2, activation=act_func_dict[actfunctype])(dense1)
        # dense3 = tf.keras.layers.Dense(n_hidden2,  kernel_regularizer='l2',activation=act_func_dict[actfunctype])(dense2)
        # Logits layer
        logits = tf.keras.layers.Dense(n_classes, activation=None)(dense2)
        predictions = tf.argmax(input=logits, axis=1)

        # Calculate Loss (for both TRAIN and EVAL modes)
        target_input = tf.placeholder(tf.int32, [None, n_classes])
        loss = tf.losses.softmax_cross_entropy(onehot_labels=target_input, logits=logits)
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopename)

        advx = pgd_attack(loss, input_layer)
    return input_layer, target_input, predictions, advx, loss, tvars


def create_implicit_image_classifier_nn(scopename, config):
    inputshape, n_classes, n_hidden1, n_hidden2, actfunctype \
        = config.dim, config.n_classes, config.n_h1, config.n_h2, config.actfunctype
    with tf.variable_scope(scopename):
        input_layer = tf.placeholder(tf.float32, [None] + inputshape, name='x')
        # pool2_flat = lenet5_convlayers(input_layer)
        pool2_flat = tf.layers.flatten(input_layer)

        with tf.GradientTape() as g:
            input_label = tf.placeholder(tf.float32, [None, 1])
            g.watch(input_label)
            concatedinput = tf.concat([pool2_flat, input_label], axis=1)
            densex1 = tf.keras.layers.Dense(n_hidden1, kernel_regularizer='l2', activation=tf.nn.tanh)(pool2_flat)
            densex2 = tf.keras.layers.Dense(n_hidden2, kernel_regularizer='l2', activation=tf.nn.tanh)(densex1)
            #densex3 = tf.keras.layers.Dense(n_hidden2, kernel_regularizer='l2', activation=tf.nn.tanh)(densex2)

            densey1 = tf.keras.layers.Dense(n_hidden1, activation=tf.nn.tanh)(input_label)
            densey2 = tf.keras.layers.Dense(n_hidden2, activation=tf.nn.tanh)(densey1)
            #densey3 = tf.keras.layers.Dense(n_hidden2, activation=tf.nn.relu)(densey2)

            fxy = tf.keras.layers.Dense(1,kernel_regularizer='l2', activation=None)(densex2)\
                  - tf.keras.layers.Dense(1, activation=None)(densey2)
        ''' compute implicit loss '''
        batch_jacobian = g.batch_jacobian(fxy, input_label)
        batch_jacobian = tf.reshape(batch_jacobian, [-1])

        implicit_loss = tf.reduce_mean(tf.square(fxy)) + tf.reduce_mean(tf.square(batch_jacobian + 2.))
        lossvec = tf.squeeze(tf.square(fxy)) + tf.square(batch_jacobian + 2.)

        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopename)
        grad_holders = [tf.placeholder(tf.float32, tvar.get_shape().as_list()) for tvar in tvars]
        return input_layer,  input_label, implicit_loss, lossvec, grad_holders, tvars

def create_linear_classification(scopename, config):
    inputshape, n_classes, label_smt = config.dim, config.n_classes, config.label_smt
    with tf.variable_scope(scopename):
        inputx = tf.placeholder(tf.float32, [None] + inputshape) \
            if isinstance(inputshape, list) else tf.placeholder(tf.float32, [None, inputshape])
        logits = tf.keras.layers.Dense(n_classes, activation=None)(tf.reshape(inputx, [-1, int(np.prod(inputshape))]))
        predictions = tf.argmax(input=logits, axis=1)
        target_input = tf.placeholder(tf.int32, [None, n_classes])
        loss = tf.losses.softmax_cross_entropy(onehot_labels=target_input, logits=logits, label_smoothing=label_smt)
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopename)
    return inputx, target_input, logits, predictions, loss, tvars


def create_minicnn(scopename, config):
    inputshape, n_classes, label_smt = config.dim, config.n_classes, config.label_smt
    # Create placeholders for input and target
    input_placeholder = tf.placeholder(tf.float32, shape=[None] + inputshape, name=scopename + 'input')
    target_placeholder = tf.placeholder(tf.int32, shape=(None, n_classes), name=scopename + 'target')
    with tf.variable_scope(scopename):
        # Define the CNN architecture
        x = input_placeholder
        for filter in [16, 32]:
            x = layers.Conv2D(filter, kernel_size=(3, 3), strides=(2, 2), activation='relu')(x)
            x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        flatten = layers.Flatten()(x)
        x = layers.Dense(units=256, activation='relu')(flatten)
        logits = layers.Dense(units=n_classes, activation=None)(x)
        # Define the prediction and loss
        prediction = tf.argmax(logits, axis=1)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=target_placeholder, logits=logits, label_smoothing=label_smt)
        mtvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopename)
    return input_placeholder, target_placeholder, logits, prediction, loss, mtvars

def create_cnn(scopename, config):
    inputshape, n_classes, label_smt = config.dim, config.n_classes, config.label_smt
    # with tf.variable_scope(scopename):
    # Create placeholders for input and target
    input_placeholder = tf.placeholder(tf.float32, shape=[None] + inputshape, name=scopename + 'input')
    target_placeholder = tf.placeholder(tf.int32, shape=(None, n_classes), name=scopename + 'target')
    with tf.variable_scope(scopename):
        # Define the CNN architecture
        x = input_placeholder
        for filter in [32, 64, 64]:
            x = layers.Conv2D(filters=filter, kernel_size=(3, 3), strides=(1, 1), activation='relu')(x)
            x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        flatten = layers.Flatten()(x)
        x = layers.Dense(units=256, activation='relu')(flatten)
        x = layers.Dense(units=256, activation='relu')(x)
        logits = layers.Dense(units=n_classes, activation=None)(x)
        # Define the prediction and loss
        prediction = tf.argmax(logits, axis=1)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=target_placeholder, logits=logits, label_smoothing=label_smt)
        mtvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopename)
    return input_placeholder, target_placeholder, logits, prediction, loss, mtvars

def residual_block(x, filters, training, kernel_size=3, stride=1):
    identity = tf.identity(x)

    # First convolution layer
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization(momentum=0.9)(x, training=training)
    x = layers.Activation('relu')(x)

    # Second convolution layer
    x = layers.Conv2D(filters, kernel_size, strides=1, padding='same')(x)

    # Shortcut connection
    if stride != 1 or identity.shape[-1] != filters:
        identity = layers.Conv2D(filters, kernel_size=1, strides=stride, padding='same')(identity)

    x = layers.add([x, identity])
    x = layers.BatchNormalization(momentum=0.9)(x, training=training)
    x = layers.Activation('relu')(x)
    return x


def create_resnet20(scopename, config):
    inputshape, n_classes, label_smt = config.dim, config.n_classes, config.label_smt
    inputshape = [None] + inputshape
    # Create placeholders for input and target
    input_placeholder = tf.placeholder(tf.float32, shape=inputshape, name=scopename+'input')
    target_placeholder = tf.placeholder(tf.int32, shape=(None, n_classes), name=scopename+'target')
    is_training_placeholder = tf.placeholder(tf.bool, shape=(), name=scopename+'is_training')
    with tf.variable_scope(scopename):
        # Define the CNN architecture using Keras layers
        x = layers.Conv2D(16, 3, strides=1, padding='same')(input_placeholder)
        x = layers.BatchNormalization(momentum=0.9)(x, training=is_training_placeholder)
        x = layers.Activation('relu')(x)

        num_blocks = 3  # ResNet-20 has 3 blocks with 6 convolutional layers each
        num_filters = 16
        for block in range(num_blocks):
            for layer in range(6):
                stride = 1
                if layer == 0 and block != 0:
                    stride = 2
                    num_filters *= 2
                x = residual_block(x,  filters=num_filters, training=is_training_placeholder, stride=stride)

        # Global average pooling and output layer
        x = layers.GlobalAveragePooling2D()(x)
        logits = layers.Dense(n_classes, activation=None)(x)

        prediction = tf.argmax(logits, axis=1)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=target_placeholder, logits=logits, label_smoothing=label_smt)

        mtvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopename)

    return input_placeholder, target_placeholder, is_training_placeholder, logits, prediction, loss, mtvars


def create_twofc_classification(inputx, n_h, actfunc, n_classes):
    if n_h > 0:
        inputx = tf.contrib.layers.fully_connected(inputx, n_h, activation_fn=actfunc)
    logits = tf.contrib.layers.fully_connected(inputx, n_classes, activation_fn=None)
    predictions = tf.argmax(input=logits, axis=1)

    # Calculate Loss (for both TRAIN and EVAL modes)
    target_input = tf.placeholder(tf.int32, [None, n_classes])
    loss = tf.losses.softmax_cross_entropy(onehot_labels=target_input, logits=logits)
    return target_input, predictions, loss


def create_image_nn_before_output_layer(config, inputx):
    inputshape, n_classes, n_hidden1, n_hidden2, actfunctype \
        = config.dim, config.n_classes, config.n_h1, config.n_h2, config.actfunctype
    # Input Layer
    pool2_flat = lenet5_convlayers(inputx)

    # Dense Layer
    dense1 = tf.contrib.layers.fully_connected(pool2_flat, n_hidden1,
                                               activation_fn=act_func_dict[actfunctype])
    dense2 = tf.contrib.layers.fully_connected(dense1, n_hidden2,
                                               activation_fn=act_func_dict[actfunctype])
    return dense2

from networks.cnns import get_lenet5_w_b_cnn, conv_net_with_fc


def create_image_classification_ltann(scopename, config):
    inputshape, n_classes, n_hidden1, n_hidden2, actfunctype \
        = config.dim, config.n_classes, config.n_h1, config.n_h2, config.actfunctype
    with tf.variable_scope(scopename):
        # Input Layer
        input_layer = tf.placeholder(tf.float32, [None] + inputshape)
        pool2_flat = lenet5_convlayers(input_layer)
        # Dense Layer
        dense1 = tf.keras.layers.Dense(n_hidden1, activation=act_func_dict[actfunctype])(pool2_flat)
        phi = tf.keras.layers.Dense(n_hidden2, activation=config.LTA.func)(dense1)

        logits = tf.keras.layers.Dense(n_classes, activation=None)(phi)
        predictions = tf.argmax(input=logits, axis=1)
        # Calculate Loss (for both TRAIN and EVAL modes)
        target_input = tf.placeholder(tf.int32, [None, n_classes])
        loss = tf.losses.softmax_cross_entropy(onehot_labels=target_input, logits=logits)
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopename)
    return input_layer, target_input, predictions, loss, phi, tvars


''' this should be a fair competitor with NN-large '''


def create_image_classification_ltamiddlenn(scopename, config):
    inputshape, n_classes, n_hidden1, n_hidden2, actfunctype \
        = config.dim, config.n_classes, config.n_h1, config.n_h2, config.actfunctype
    with tf.variable_scope(scopename):
        # Input Layer
        input_layer = tf.placeholder(tf.float32, [None] + inputshape)
        pool2_flat = lenet5_convlayers(input_layer)
        # sparse Layer
        phi1 = tf.contrib.layers.fully_connected(pool2_flat, n_hidden1, activation_fn=config.LTA.func)
        phi2 = tf.contrib.layers.fully_connected(phi1, n_hidden2, activation_fn=config.LTA.func)
        # Logits layer
        logits = tf.contrib.layers.fully_connected(phi2, n_classes,
                                                   weights_regularizer=tf.contrib.layers.l2_regularizer(config.l2reg),
                                                   activation_fn=None)
        predictions = tf.argmax(input=logits, axis=1)

        # Calculate Loss (for both TRAIN and EVAL modes)
        target_input = tf.placeholder(tf.int32, [None, n_classes])
        loss = tf.losses.softmax_cross_entropy(onehot_labels=target_input, logits=logits)
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopename)
    return input_layer, target_input, predictions, loss, phi1, phi2, tvars