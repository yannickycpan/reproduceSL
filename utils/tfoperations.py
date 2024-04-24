import tensorflow as tf
import numpy as np


def minrelu(x):
    return tf.minimum(x, 0)


def tanhminrelu(x):
    return tf.nn.tanh(minrelu(x))


def leaky_relu(x):
    return tf.maximum(0.001*x, x)


def low_relu(x):
    return tf.maximum(-1., x)


act_func_dict = {'tanh': tf.nn.tanh, 'linear': lambda x: x, 'relu': tf.nn.relu, 'sigmoid': tf.nn.sigmoid,
                 'minrelu': minrelu, 'tanhminrelu': tanhminrelu, 'leaky_relu': leaky_relu, 'low_relu': low_relu}


def sigmoid_cross_entropy(logits, labels):
    return tf.nn.relu(logits) - logits * labels + tf.log(1. + tf.exp(-tf.abs(logits)))

''' the jacobian of y w.r.t. x '''
''' x can be an image, y is a vector '''


def compute_jacobian(y, x, xtotaldim):
    print('----------------', tf.gradients(y, x)[0].shape)
    return tf.stack([tf.layers.flatten(tf.gradients(y[:, idx], x)[0])
              for idx in range(xtotaldim)], axis=0)


def compute_jacobian_plus_gradient(y, x, xtotaldim):
    # print('----------------', tf.gradients(y[0], x)[0].shape)
    y = tf.expand_dims(y, 0)
    x = tf.expand_dims(x, 0)
    return tf.square(tf.stack([tf.gradients(y[:, idx], x)[0]
              for idx in range(xtotaldim)], axis=0) + tf.eye(xtotaldim))


def compute_hessian(y, x, xdim):
    y_wrt_x = tf.gradients(y, x)[0]
    hessian = tf.stack(
        [tf.reshape(tf.gradients(y_wrt_x[:, idx], [x])[0], [-1])
         for idx in range(xdim)], axis=0)
    return hessian


def create_two_layer_fc(x, n_hidden1, n_hidden2, actfunc):
    hidden1 = tf.contrib.layers.fully_connected(x, n_hidden1, activation_fn=actfunc)
    hidden2 = tf.contrib.layers.fully_connected(hidden1, n_hidden2, activation_fn=actfunc)
    return hidden2


def extract_3d_tensor(tensor, indexes):
    rowinds = tf.range(0, tf.cast(tf.shape(tensor)[0], tf.int64), 1)
    ind_nd = tf.concat([tf.reshape(rowinds, [-1, 1]), tf.reshape(indexes, [-1, 1])], axis=1)
    extracted = tf.gather_nd(tensor, ind_nd)
    return extracted


'''extract the indexes elements along the second dim of tensor'''
def extract_2d_tensor(tensor, indexes):
    one_hot = tf.one_hot(indexes, tf.shape(tensor)[1], 1.0, 0.0)
    extracted = tf.reduce_sum(tensor * one_hot, axis=1)
    return extracted


def get_tvars_count(trainvars):
    n_total_vars = 0
    for tvar in trainvars:
        n_total_vars += int(np.prod(tvar.shape))
    return n_total_vars

def update_target_nn_assign(tar_tvars, tvars):
    target_params_update = [tf.assign(tar_tvars[idx],  tvars[idx]) for idx in range(len(tvars))]
    return target_params_update

def huber_loss(mat, kaap):
    condition = tf.cast((tf.abs(mat) <= kaap), tf.float32)
    huberloss_mat = 0.5 * tf.square(mat) * condition + kaap * (tf.abs(mat) - 0.5 * kaap) * (1.0 - condition)
    return huberloss_mat

def listshape2vec(list_tvars):
    return tf.concat([tf.reshape(varpart, [-1]) for varpart in list_tvars], axis=0)

def reshape2vars(vectorholder, sizelist, shapelist):
    shapedholders = []
    count = 0
    for idx in range(len(shapelist)):
        shapedholders.append(tf.reshape(vectorholder[count:count + sizelist[idx]], shapelist[idx]))
        count += sizelist[idx]
    return shapedholders


def limit_approx_sign(x, epsilon=1e-5):
    return x/tf.sqrt(tf.square(x) + tf.square(epsilon))


def tanh_approx_sign(x, scalor=100.0):
    return tf.nn.tanh(scalor*x)


''' this is the leaky operation in the paper '''
def indicator_approx_sign(x, eta):
    return tf.cast(x <= eta, tf.float32)*x + tf.cast(x > eta, tf.float32)


def create_tile_coding_complex(shoot, strength, c_list, delta_list, sparsed_dim, sgn_factor=100.):
    k = len(c_list)
    shoot = tf.reshape(shoot, [-1, k, 1])
    strength = tf.reshape(strength, [-1, k, 1])
    #print(' test broadcasting :: ----------------- ', c_list[0].shape, shoot[:, 0, :].shape, (c_list[0]-shoot[:, 0, :]).shape)
    onehot = [(1.0 - tanh_approx_sign(tf.nn.relu(c_list[i]-shoot[:, i, :])
              + tf.nn.relu(shoot[:, i, :]-delta_list[i]-c_list[i]), sgn_factor))*strength[:, i, :]
              for i in range(k)]
    ''' after concat, the shape should be b * (d_1 + ... + d_k)'''
    onehot = tf.concat(onehot, axis=1)
    print(' after concat the onehot dimension is :: ', onehot.shape)
    onehot = tf.reshape(onehot, [-1, sparsed_dim])
    # return a sparse vector with only k nonzero entries
    ''' cast all values close to zero as zero '''
    # onehot = tf.cast(tf.abs(onehot) > 0.1, tf.float32) * onehot
    return onehot


''' c has shape (tilinglength=k, ), will be broadcasted to (minibatchsize, d, k) shape '''
''' LTA maps d features to dk features '''
''' the notation is now consistent with the paper '''


def LTA(shoot, strength, c, delta, eta, d, k):
    x = tf.reshape(shoot, [-1, d, 1])
    strength = tf.reshape(strength, [-1, d, 1]) \
        if not isinstance(strength, float) else strength
    onehot = (1.0 - indicator_approx_sign(tf.nn.relu(c - x) + tf.nn.relu(x - delta - c), eta)) * strength
    onehot = tf.reshape(onehot, [-1, int(d * k)])
    print(' after LTA processing the onehot dimension is :: ', onehot.shape)
    # return a sparse vector with only d nonzero entries
    return onehot


''' multitiling in fact is general to cover the LTA with one tiling (above);
    delta is the same across tilings, though it is easy to extend to different deltas '''


def LTA_multitilings(shoot, strength, c_list, delta, eta, d, k):
    x = tf.reshape(shoot, [-1, d, 1])
    strength = tf.reshape(strength, [-1, d, 1]) \
        if not isinstance(strength, float) else strength
    ''' for each tiling, operates on all of the input units '''
    onehots = []
    for c in c_list:
        onehots.append(tf.reshape((1.0
                                   - indicator_approx_sign(tf.nn.relu(c - x) + tf.nn.relu(x - delta - c),
                                                           eta)) * strength, [-1, k]))
    onehot = tf.reshape(tf.concat(onehots, axis=1), [-1, int(len(c_list) * d * k)])
    print(' after LTA multitiling processing the onehot dimension is :: ', onehot.shape)
    return onehot


''' NOT IN USE '''
def LTA_multitilings_old(shoot, strength, multiple_clists, delta_list, sparsed_dim):
    k = len(multiple_clists[0])
    shoot = tf.reshape(shoot, [-1, k, 1])
    strength = tf.reshape(strength, [-1, k, 1])
    onehots = []
    for c_list in multiple_clists:
        onehots.append(tf.reshape(tf.concat([(1.0 - indicator_approx_sign(tf.nn.relu(c_list[i] - shoot[:, i, :])
                                      + tf.nn.relu(shoot[:, i, :] - delta_list[i] - c_list[i]), delta_list[i]))
              * strength[:, i, :] for i in range(k)], axis=1), [-1, sparsed_dim]))
    onehot = tf.concat(onehots, axis=1)
    print(' after concat the onehot dimension is :: ', onehot.shape)
    return onehot


''' NOT IN USE :: this is original version, allows different input unit to have different tilings '''
def create_tile_coding_complex_constraint_indicator(shoot, strength, c_list, delta_list, sparsed_dim, sgn_factor, tiles_bound, tanh_factor):
    k = len(c_list)
    shoot = tf.reshape(shoot, [-1, k, 1])
    strength = tf.reshape(strength, [-1, k, 1])
    # print(' test broadcasting :: ----------------- ', c_list[0].shape, shoot[:, 0, :].shape, (c_list[0]-shoot[:, 0, :]).shape)
    onehot = [(1.0 - indicator_approx_sign(tf.nn.relu(c_list[i] - shoot[:, i, :])
                                      + tf.nn.relu(shoot[:, i, :] - delta_list[i] - c_list[i]), delta_list[i],
                                      tanh_factor))
              * strength[:, i, :] for i in range(k)]
    ''' after concat, the shape should be b * (d_1 + ... + d_k)'''
    onehot = tf.concat(onehot, axis=1)
    print(' after concat the onehot dimension is :: ', onehot.shape)
    onehot = tf.reshape(onehot, [-1, sparsed_dim])
    return onehot


def create_tile_coding_complex_constraint_tanh(shoot, strength, c_list, delta_list, sparsed_dim, sgn_factor, tiles_bound, tanh_factor):
    k = len(c_list)
    shoot = tf.reshape(shoot, [-1, k, 1])
    strength = tf.reshape(strength, [-1, k, 1])
    onehot = [(1.0 - tanh_approx_sign(tf.nn.relu(c_list[i] - shoot[:, i, :])
                                      + tf.nn.relu(shoot[:, i, :] - delta_list[i] - c_list[i]),
                                      tanh_factor))
              * strength[:, i, :] for i in range(k)]
    ''' after concat, the shape should be b * (d_1 + ... + d_k)'''
    onehot = tf.concat(onehot, axis=1)
    print(' after concat the onehot dimension is :: ', onehot.shape)
    onehot = tf.reshape(onehot, [-1, sparsed_dim])
    return onehot


def create_rbf_complex_constraint(shoot, c_list, sparsed_dim, bandwidth):
    k = len(c_list)
    shoot = tf.reshape(shoot, [-1, k, 1])
    ''' now the sgn factor is the bandlimit '''
    onehot = [tf.exp(-bandwidth*tf.square(c_list[i] - shoot[:, i, :])) for i in range(k)]
    ''' after concat, the shape should be b * (d_1 + ... + d_k)'''
    onehot = tf.concat(onehot, axis=1)
    print(' after concat the onehot dimension is :: ', onehot.shape)
    onehot = tf.reshape(onehot, [-1, sparsed_dim])
    return onehot


def vae_loss(img_input, img_out, mean_, std_dev, img_width):
    flat_output = tf.reshape(img_out, [-1, img_width * img_width])
    flat_input = tf.reshape(img_input, [-1, img_width * img_width])

    img_loss = tf.reduce_sum(flat_input * -tf.log(flat_output) + (1 - flat_input) * -tf.log(1 - flat_output), 1)
    latent_loss = 0.5 * tf.reduce_sum(tf.square(mean_) + tf.square(std_dev) - tf.log(tf.square(std_dev)) - 1, 1)
    loss = tf.reduce_mean(img_loss + latent_loss)
    return loss

def ae_loss(img_input, img_out, img_width):
    flat_output = tf.reshape(img_out, [-1, img_width * img_width])
    flat_input = tf.reshape(img_input, [-1, img_width * img_width])
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(flat_output - flat_input), axis=1))
    return loss
