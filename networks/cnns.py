import tensorflow as tf

def conv2d(x, W, b, strides=1, actfunc=tf.nn.relu):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return actfunc(x)


def conv_net_with_fc(x, weights, biases, actfunc=tf.nn.relu):
    conv1 = conv2d(x, weights['wc1'], biases['bc1'], 1, actfunc)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = conv2d(pool1, weights['wc2'], biases['bc2'], 1, actfunc)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    fc1 = tf.add(tf.matmul(tf.layers.flatten(pool2), weights['wd1']), biases['bd1'])
    fc1 = actfunc(fc1)
    return fc1


def get_lenet5_w_b_cnn(nchannels, flattened_n, n_h1):
    ks = 5
    n_filters_1 = 6
    n_filters_2 = 16
    #def size_linear_unit(size, kernel_size=ks, stride=1):
    #    return (size - (kernel_size - 1) - 1) // stride + 1
    weights = {
        'wc1': tf.get_variable('W0', shape=(ks, ks, nchannels, n_filters_1)),
        'wc2': tf.get_variable('W1', shape=(ks, ks, n_filters_1, n_filters_2)),
        'wd1': tf.get_variable('W2', shape=(flattened_n, n_h1))
    }
    biases = {
        'bc1': tf.get_variable('B0', shape=(n_filters_1)),
        'bc2': tf.get_variable('B1', shape=(n_filters_2)),
        'bd1': tf.get_variable('B2', shape=(n_h1))
    }
    return weights, biases


def lenet5_convlayers(input_layer, actfn = tf.nn.relu):
    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 32, 32, 1]
    # Output Tensor Shape: [batch_size, 28, 28, 6]
    conv1 = tf.keras.layers.Conv2D(
        # inputs=input_layer,
        filters=6,
        kernel_size=[5, 5],
        padding="valid",
        activation=actfn)(input_layer)
    # output is 14, 14, 6, valid padding by default
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid')(conv1)

    # Convolutional Layer #2
    # Input Tensor Shape: [batch_size, 14, 14, 6]
    # Output Tensor Shape: [batch_size, 10, 10, 16]
    conv2 = tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=[5, 5],
        padding="valid",
        activation=actfn)(pool1)
    # Pooling Layer #2, output becomes [5, 5, 16]
    # pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv2)

    # Flatten tensor into a batch of vectors
    #pool2_flat = tf.reshape(pool2, [-1, 5 * 5 * 16])
    pool2_flat = tf.layers.flatten(pool2)
    return pool2_flat


def lenet5_deconvlayers(input_layer, actfn = tf.nn.relu):
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
