import numpy as np
import tensorflow as tf

def weight_init(n_prev, n_next):
    winit = np.random.normal(np.linspace(-1.0 / n_prev, 1.0 / n_prev, n_next).reshape((n_next, 1)),
                             1.0 / np.sqrt(n_prev),
                             (n_next, n_prev)).astype(np.float32)
    print(' the mean value of each column is ', np.mean(winit.T, axis=0))
    return winit.T

def compute_activation_overlap(phix, phixp):
    act_overlap = np.mean(np.sum((phix > 0) * (phixp > 0), axis=1)/phix.shape[1])
    return act_overlap


def compute_instance_sparsity(phix):
    sparse_prop = np.mean(np.sum((np.abs(phix) > 1e-5), axis=1)/float(phix.shape[1]))
    return sparse_prop


def compute_activation_freq(phi):
    return np.sum(phi > 0, axis=0)/float(phi.shape[0])


def get_tilings_tanh(n_tilings, tile_delta=0.01):
    npdelta = np.array([tile_delta]*n_tilings)
    #npdelta = np.array([0.2, 0.5])
    print('np delta is :: ', npdelta)
    startc = np.array([-1.] * n_tilings)
    #startc = np.arange(-1., n_tilings*tile_delta + tile_delta, tile_delta)
    print('the starting loc of each tiling is :: =================================== ', startc)
    np_c = [np.arange(startc[i], startc[i] + 2., npdelta[i]).astype(np.float32) for i in range(n_tilings)]
    sparse_dim = int(np.sum([arr.shape[0] for arr in np_c]))
    return npdelta, np_c, sparse_dim


def get_tilings_relu(n_tilings, tile_delta=0.01, tile_bound=100.):
    npdelta = np.array([tile_delta]*n_tilings)
    print('np delta is :: ', npdelta)
    startc = np.arange(0, n_tilings*tile_delta + tile_delta, tile_delta)
    np_c = [np.arange(startc[i], startc[i] + tile_bound, npdelta[i]).astype(np.float32) for i in range(n_tilings)]
    sparse_dim = int(np.sum([arr.shape[0] for arr in np_c]))
    return npdelta, np_c, sparse_dim


def get_tilings_linear(n_tilings, tile_delta=0.1, tiling_bound=10.):
    npdelta = np.array([tile_delta]*n_tilings)
    #npdelta = np.array([2, 5])
    print('np delta is :: ', npdelta)
    #startc = np.arange(-tiling_bound+tile_delta, n_tilings*tile_delta + 2.*tile_delta, tile_delta)
    startc = np.array([-tiling_bound]*n_tilings)
    np_c = [np.arange(startc[i], startc[i] + 2.*tiling_bound, npdelta[i]).astype(np.float32) for i in range(n_tilings)]
    sparse_dim = int(np.sum([arr.shape[0] for arr in np_c]))
    return npdelta, np_c, sparse_dim


def get_numpy_tc_setting(n_tilings, tile_delta, latentacttype, tile_bound):
    if latentacttype == 0:
        npdelta, np_c, sparse_dim = get_tilings_tanh(n_tilings, tile_delta)
        tile_bound = 1.0
    elif latentacttype == 1:
        npdelta, np_c, sparse_dim = get_tilings_linear(n_tilings, tile_delta, tile_bound)
    else:
        npdelta, np_c, sparse_dim = get_tilings_relu(n_tilings, tile_delta, tile_bound)
    delta_list = [tf.constant(v.astype(np.float32)) for v in npdelta]
    c_list = [tf.constant(np_c[i]) for i in range(n_tilings)]
    return delta_list, c_list, tile_bound, sparse_dim


''' rigorously the first param n_tilings is the number of hidden units in the last hidden layer '''
''' n_tile is k in the paper '''
''' assume input range is -tile_bound to tile_bound '''


def get_tilings_offset(n_tilings, n_tile, input_min, input_max):
    maxoffset = (input_max - input_min)/n_tile
    tiling_length = input_max - input_min + maxoffset
    tile_delta = tiling_length/n_tile

    print(' immediately computed tile delta is ======================= ', tile_delta)
    if n_tilings == 1:
        one_c = np.linspace(input_min-maxoffset/2., input_max+maxoffset/2., n_tile, endpoint=False).astype(np.float32)
        c_list = [tf.constant(one_c)]
        tile_delta = one_c[1] - one_c[0]
        print(' second time computed tile delta is ==================== ', tile_delta, one_c.shape, one_c[2]-one_c[1])
        print(' the generated list is ================================   ', one_c)
        return c_list, tile_delta, input_min-maxoffset/2., input_max+maxoffset/2.
    startc = input_min - np.random.uniform(0, maxoffset, n_tilings)
    print(' the offset values are :: ========================================== ', startc)
    c_list = []
    for n in range(n_tilings):
        one_c = np.linspace(startc[n], startc[n] + tiling_length, n_tile, endpoint=False).astype(np.float32)
        c_list.append(tf.constant(one_c.copy().astype(np.float32)))
        print(' second time computed tile delta is ==================== ', tile_delta, one_c.shape, one_c[2]-one_c[1])
        print(one_c)
    tiling_low_bound = np.min(startc) - maxoffset
    tiling_up_bound = np.max(startc) + tiling_length
    return c_list, tile_delta, tiling_low_bound, tiling_up_bound


def get_tilings(n_tilings, n_tile, input_min, input_max):
    tiling_length = input_max - input_min
    tile_delta = (input_max - input_min)/n_tile
    print(' immediately computed tile delta is ======================= ', tile_delta)
    if n_tilings == 1:
        one_c = np.linspace(input_min, input_max, n_tile, endpoint=False).astype(np.float32)
        c_list = [tf.constant(one_c)]
        tile_delta = one_c[1] - one_c[0]
        print(' second time computed tile delta is ==================== ', tile_delta, one_c.shape, one_c[2]-one_c[1])
        print(' the generated list is ================================   ', one_c)
        return c_list, tile_delta, input_min, input_max
    maxoffset = (input_max - input_min)/n_tile
    startc = input_min - np.random.uniform(0, maxoffset, n_tilings)
    print(' the offset values are :: ========================================== ', startc)
    c_list = []
    for n in range(n_tilings):
        one_c = np.linspace(startc[n], startc[n] + tiling_length, n_tile, endpoint=False).astype(np.float32)
        c_list.append(tf.constant(one_c.copy().astype(np.float32)))
        print(' second time computed tile delta is ==================== ', tile_delta, one_c.shape, one_c[2]-one_c[1])
        print(one_c)
    tiling_low_bound = np.min(startc) - maxoffset
    tiling_up_bound = np.max(startc) + tiling_length
    return c_list, tile_delta, tiling_low_bound, tiling_up_bound