import tensorflow as tf
import numpy as np


class LTAConfiguration(object):
    default_attributes = {'n_tiles': 20, 'n_tilings': 1, 'sparse_dim': None,
                          'test_tiling': False, 'lta_input_max': 1.0, 'lta_input_min': -1.0, 'lta_eta': 0.1,
                          'outofbound_reg': 0.0, 'self_strength': False, 'extra_strength': False,
                          'individual_tiling': False, 'train_bound': False, 'similarity': 'IPlusEta',
                          'actfunctypeLTA': 'linear',  'actfunctypeLTAstrength': 'linear',
                          'max_scalor': 10.0}

    def __init__(self, configdict):
        for key in configdict:
            if key in self.default_attributes:
                setattr(self, key, configdict[key])

        if not hasattr(self, 'lta_input_min'):
            self.lta_input_min = -self.lta_input_max
        if not hasattr(self, 'lta_eta'):
            self.lta_eta = (self.lta_input_max - self.lta_input_min)/self.n_tiles

        for key in self.default_attributes:
            if not hasattr(self, key):
                setattr(self, key, self.default_attributes[key])
        if self.n_tilings > 1:
            ''' if multi-tiling, use default setting, lta_input_max should be a list '''
            return


''' 
similarity measures between a scalar and a vector, return a vector;
or a vector and matrix, the return value is input-dependent,
be careful about broadcasting 
'''


class similarity_measures(object):

    @staticmethod
    def sum_relu(c, x, delta):
        return tf.nn.relu(c - x) + tf.nn.relu(x - delta - c)

    @staticmethod
    def Iplus_eta(x, eta):
        if eta == 0:
            return tf.math.sign(x)
        return tf.cast(x <= eta, tf.float32) * x/eta + tf.cast(x > eta, tf.float32)

    @staticmethod
    def one_minus_Iplus_eta(x, c, delta, eta):
        return 1.0 - similarity_measures.Iplus_eta(similarity_measures.sum_relu(c, x, delta), eta)

    @staticmethod
    def one_minus_tanh(x, c, delta, eta):
        return 1.0 - tf.nn.tanh(similarity_measures.sum_relu(c, x, delta) * eta)

    @staticmethod
    def rbf(x, c, delta, eta):
        return tf.exp(-tf.square(c - x) / eta)


class LTA(object):

    act_func_dict = {'tanh': tf.nn.tanh, 'linear': lambda x: x,
                     'relu': tf.nn.relu, 'sigmoid': tf.nn.sigmoid, 'clip': None, 'sin': tf.math.sin, 'cos': tf.math.cos}

    similarity_dict = {'IPlusEta': similarity_measures.one_minus_Iplus_eta, 'RBF': similarity_measures.rbf,
                       'Tanh': similarity_measures.one_minus_tanh}

    def __init__(self, params, actornn=''):
        config = LTAConfiguration(params)
        self.config = config
        ''' rewrite the clip activation function '''
        self.act_func_dict['clip'] = lambda x: tf.clip_by_value(x, config.lta_input_min, config.lta_input_max)
        self.similarity = self.similarity_dict[config.similarity]
        ''' set tilings, tiles '''
        self.c_mat, self.tile_delta_vector \
                = self.get_multi_tilings(config.n_tilings, config.n_tiles)

        if config.train_bound:
            self.v = tf.Variable(0.)
            self.lta_bound = tf.exp(self.v)
            self.bound_w = tf.get_variable('ltaboundw' + actornn, shape=(params['n_h1'], 1))
            self.func = self.LTA_func_var

            self.bd_vars = tf.get_variable('Wf', shape=(32, 1))
            self.lta_indv_bound = tf.exp(self.bd_vars)
            # self.func = self.LTA_func_var_add_to_input
        elif config.n_tilings > 1:
            self.func = self.LTA_func_w
        else:
            self.func = self.LTA_func
        print(' lta_eta, n_tilings, and n_tiles :: ===================================================== ',
              config.lta_eta, config.n_tilings, config.n_tiles, config.lta_input_min, config.lta_input_max)

    def Iplus_eta(self, x, eta):
        if eta == 0:
            return tf.math.sign(x)
        #return tf.cast(x <= eta, tf.float32) * x/eta + tf.cast(x > eta, tf.float32)
        # return tf.math.minimum(x/eta, 1.)
        return tf.nn.relu(eta - x) * x/eta + tf.cast(x > eta, tf.float32)

    def _sum_relu(self, c, x, delta):
        return tf.nn.relu(c - x) + tf.nn.relu(x - delta - c)

    def get_bound(self, sess):
        return sess.run(self.lta_bound)

    def get_state_dependent_bound(self, sess, feed):
        if self.state_lta_bound is None:
            return 0.
        # print(' the current eta w value is ============================ ', sess.run(tf.nn.sigmoid(self.eta_w)))
        return sess.run(self.state_lta_bound, feed_dict=feed)

    ''' for each tiling, operates on all of the input units; if rawinput is None, strenght is one '''
    def get_sparse_vector(self, input, rawinput, n_tiles, tile_delta, lta_eta, c_list):
        d = int(input.shape.as_list()[1])
        k = int(n_tiles)
        x = tf.reshape(input, [-1, d, 1])
        onehots = []
        for c in c_list:
            onehots.append(
                tf.reshape((1.0 - self.Iplus_eta(self._sum_relu(c, x, tile_delta),
                                                            lta_eta)), [-1, k])
            )
        onehot = tf.reshape(tf.concat(onehots, axis=1), [-1, int(len(c_list) * d * k)])
        return onehot

    def check_clist(self):
        return [tf.convert_to_tensor([-self.lta_bound + i * 2. * self.lta_bound / self.config.n_tiles
                                      for i in range(self.config.n_tiles)])]

    ''' the bound should be defined as the input-dependent, hence the sparsity would be also
        input-dependent. Maybe input plus some value, input scaled by some value, 
        or some more complicated linear relation to input  '''
    def LTA_func_var_input_dependent(self, rawinput, bounds):
        """ this activation function decides if we should preprocess before feeding into LTA function """
        input = self.act_func_dict[self.actfunctypeLTA](rawinput)
        ''' the bound is state dependent, b * 1 '''
        # bounds = tf.exp(tf.matmul(input, self.bound_w))
        self.state_lta_bound = bounds if self.state_lta_bound is None else self.state_lta_bound
        delta = 2.*bounds/self.config.n_tiles
        ''' cmat is b-by-ntiles, delta is b-by-1 '''
        c_mat = tf.concat([-bounds + i * delta for i in range(self.config.n_tiles)], axis=1)
        c_mat = tf.reshape(c_mat, (-1, 1, self.config.n_tiles))
        delta = tf.reshape(delta, (-1, 1, 1))
        d = int(input.shape.as_list()[1])
        x = tf.reshape(input, [-1, d, 1])
        ''' 
        each row in the c_mat is a tiling, for each sample's each hidden unit,
        apply one tiling, 4. * tf.stop_gradient(delta) 
        '''
        onehots = 1.0 - self.Iplus_eta(self._sum_relu(c_mat, x, delta), 4.0 * delta)
        onehot = tf.reshape(onehots, [-1, int(d * self.config.n_tiles)])
        print(' after LTA processing the onehot dimension is :: ', onehot.shape)
        return onehot

    def LTA_func_var_add_to_input(self, rawinput):
        """ this activation function decides if we should preprocess before feeding into LTA function """
        input = self.act_func_dict[self.config.actfunctypeLTA](rawinput)
        d = int(input.shape.as_list()[1])
        x = tf.reshape(input, [-1, d, 1])
        ''' bounds on the right is d-by-1 '''
        bounds = tf.stop_gradient(tf.abs(input)) + tf.reshape(tf.exp(self.bound_w), [1, d])
        ''' bounds on the right is b-by-1 or b-by-d '''
        # bounds = tf.abs(input) + bounds
        # self.state_lta_bound = bounds if self.state_lta_bound is None else self.state_lta_bound
        bounds = tf.reshape(bounds, [-1, d, 1])
        delta = 2.*bounds/self.config.n_tiles
        # delta = self.lta_eta
        ''' cmat is b-by-d-by-ntiles '''
        c_mat = tf.concat([-bounds + i * delta for i in range(self.config.n_tiles)], axis=2)
        print(' the shape of cmat is ========================================== ', c_mat.shape)
        # eta = tf.nn.sigmoid(self.eta_w) * 4. * delta
        onehots = 1.0 - self.Iplus_eta(self._sum_relu(c_mat, x, delta), self.config.lta_eta)
        # onehots = self.similarity(x, self.c_mat, self.tile_delta_vector, self.config.lta_eta)
        onehot = tf.reshape(onehots, [-1, int(d * self.config.n_tiles)])
        print(' after LTA processing the onehot dimension is :: =========================== ', onehot.shape)
        return onehot

    ''' the above should be more inclusive than this one, TODO: double check '''
    def LTA_func_var(self, rawinput):
        """ this activation function decides if we should preprocess before feeding into LTA function """
        input = self.act_func_dict[self.config.actfunctypeLTA](rawinput)
        """ range function is not differentiable """
        # c_list = [tf.range(-self.lta_bound, self.lta_bound, delta=2.*self.lta_bound/self.n_tiles, dtype=tf.float32)]
        delta = 2.*self.lta_bound/self.config.n_tiles
        c_list = [tf.convert_to_tensor([-self.lta_bound + i*delta for i in range(self.config.n_tiles)])]
        # delta = tf.stop_gradient(2.*self.lta_bound/self.n_tiles)
        onehot = self.get_sparse_vector(input, rawinput, self.config.n_tiles, delta, self.config.lta_eta, c_list)
        print(' after LTA processing the onehot dimension is :: ', onehot.shape)
        return onehot

    ''' 
    get multiple tiling vectors, now lta_input_max is a list of upper bounds;
    need to study what tilings should be used; 
    TODO: does it work if it is just one tiling? 
    '''
    def get_multi_tilings(self, n_tilings, n_tile):
        if n_tilings == 1 and not self.config.individual_tiling:
            input_max_list = [self.config.lta_input_max]
        elif self.config.individual_tiling:
            input_max_list = np.random.choice(self.config.lta_input_max, n_tilings) + [self.config.lta_input_max]
        else:
            # maxoffset = n_tilings * (self.config.lta_input_max - self.config.lta_input_min) / n_tile
            #input_max_list = #self.config.lta_input_max + np.random.uniform(0, maxoffset, n_tilings-1) \
            #                 + [self.config.lta_input_max]
            input_max_list = list(np.abs(np.random.uniform(0.1, 5., n_tilings))) #+ [self.config.lta_input_max]
            print(' the generated maxoffset list is :: ================================== ', input_max_list)
        c_list = []
        tile_delta_list = []
        for n in range(n_tilings):
            ind = n % len(input_max_list)
            one_c = np.linspace(-input_max_list[ind], input_max_list[ind], n_tile, endpoint=False).astype(np.float32)
            c_list.append(tf.constant(one_c.copy().astype(np.float32).reshape((-1, n_tile))))
            tile_delta_list.append((one_c[1]-one_c[0]))
        c_mat = tf.concat(c_list, axis=0)
        tile_delta_vector = tf.reshape(tf.constant(np.array(tile_delta_list).astype(np.float32)), [n_tilings, 1])
        return c_mat, tile_delta_vector

    ''' 
    rawinput has shape (minibatchsize, # of hidden units)
    for example, rawinput = h W, h is the previous layer's output,
    W is the weight matrix in the current hidden layer whose activation is LTA.
    If h is a-by-b, W is b-by-c, then rawinput is a-by-c. If LTA has n_tilings and n_tiles, 
    then the output has shape (a, c * n_tilings * n_tiles). 
    '''
    def LTA_func(self, rawinput):
        input = self.act_func_dict[self.config.actfunctypeLTA](rawinput)
        d = int(input.shape.as_list()[1])
        if self.config.n_tilings > 1:
            x = tf.reshape(input, [-1, d, 1, 1])
        else:
            x = tf.reshape(input, [-1, d, 1])
        ''' each row in the c_mat is a tiling, for each sample's each hidden unit, apply all those tilings  '''
        onehots = self.similarity(x, self.c_mat, self.tile_delta_vector, self.config.lta_eta)
        onehot = tf.reshape(onehots, [-1, int(d * self.config.n_tiles * self.config.n_tilings)])
        print(' after LTA processing the onehot dimension is :: ', onehot.shape)
        return onehot

    def LTA_func_w(self, rawinput):
        input = self.act_func_dict[self.config.actfunctypeLTA](rawinput)
        n, d = int(input.shape.as_list()[0]), int(input.shape.as_list()[1])
        x = tf.reshape(input, [-1, d, 1])
        ''' each row in the c_mat is a tiling, for each sample's each hidden unit, apply all those tilings  '''
        if self.config.n_tilings > 1:
            # onehots = self.similarity(x, self.c_mat[:,None,:], self.tile_delta_vector[:,None,:], self.config.lta_eta)
            onehots = self.similarity(x, self.c_mat[:, None, :], self.tile_delta_vector[:, None, :],
                                      2. * self.tile_delta_vector[:, None, :])
        else:
            onehots = self.similarity(x, self.c_mat, self.tile_delta_vector, self.config.lta_eta)
        onehot = tf.reshape(onehots, [n, int(d * self.config.n_tiles)])
        print(' after LTA processing the onehot dimension is :: ------------------------- ', onehot.shape)
        return onehot

    def LTA_func_train_bound(self, rawinput):
        """ this activation function decides if we should preprocess before feeding into LTA function """
        input = self.act_func_dict[self.config.actfunctypeLTA](rawinput)
        n = int(input.shape.as_list()[0])
        d = int(input.shape.as_list()[1])
        x = tf.reshape(input, [-1, d, 1])
        ''' bounds is n-by-1 '''
        bounds = tf.reshape(self.lta_indv_bound, [n, 1])
        delta = 2.*bounds/self.config.n_tiles
        ''' cmat is n-by-ntiles '''
        c_mat = tf.concat([-bounds + i * delta for i in range(self.config.n_tiles)], axis=1)
        print(' the shape of cmat is ========================================== ', c_mat.shape)

        onehots = self.similarity(x, c_mat[:, None, :], delta[:, None, :], 2. * delta[:, None, :])
        onehot = tf.reshape(onehots, [-1, int(d * self.config.n_tiles)])
        print(' after LTA processing the onehot dimension is :: =========================== ', onehot.shape)
        return onehot


    def LTA_func_eta(self, rawinput, eta):
        input = self.act_func_dict[self.config.actfunctypeLTA](rawinput)
        d = int(input.shape.as_list()[1])
        if self.config.n_tilings > 1:
            x = tf.reshape(input, [-1, d, 1, 1])
        else:
            x = tf.reshape(input, [-1, d, 1])
        ''' each row in the c_mat is a tiling, for each sample's each hidden unit, apply all those tilings  '''
        onehots = self.similarity(x, self.c_mat, self.tile_delta_vector, eta)
        onehot = tf.reshape(onehots, [-1, int(d * self.config.n_tiles * self.config.n_tilings)])
        print(' after LTA processing the onehot dimension is :: ', onehot.shape)
        return onehot

    def LTA_func_rbf(self, rawinput):
        input = self.act_func_dict[self.config.actfunctypeLTA](rawinput)
        d = int(input.shape.as_list()[1])
        if self.config.n_tilings > 1:
            x = tf.reshape(input, [-1, d, 1, 1])
        else:
            x = tf.reshape(input, [-1, d, 1])
        ''' each row in the c_mat is a tiling, for each sample's each hidden unit, apply all those tilings  '''
        onehots = similarity_measures.rbf(x, self.c_mat, self.tile_delta_vector, self.config.lta_eta * 2.)
        onehot = tf.reshape(onehots, [-1, int(d * self.config.n_tiles * self.config.n_tilings)])
        print(' after LTA processing the onehot dimension is :: ', onehot.shape)
        return onehot


if __name__ == '__main__':
    n_tiles = 10
    bound = 1.0
    params = {'n_tiles': n_tiles, 'n_tilings': 1, 'sparse_dim': None, 'lta_eta': 2.*bound/n_tiles,
              'individual_tiling': False, 'similarity': 'IPlusEta',
                          'test_tiling': False, 'lta_input_max': bound, 'train_bound': False,
                          'outofbound_reg': 0.0, 'self_strength': False, 'extra_strength': False,
                          'actfunctypeLTA': 'linear',  'actfunctypeLTAstrength': 'linear',
                          'max_scalor': 10.0, 'n_h2': 1}
    lta = LTA(params)
    # lta.sess.run(tf.global_variables_initializer())
    # tf.Session().run(tf.global_variables_initializer())

    mymat = np.array([[0.22, 0.32], [8.9, 0.8]])
    mymat = np.array([[-3.2], [0.22], [0.32], [1.0], [2.0]])
    mymat = np.arange(-bound, bound, 0.1).reshape((-1, 1))
    # mymat = np.arange(-10. * bound, 10. * bound, 0.01).reshape((-1, 1))

    testinput = tf.constant(mymat.astype(np.float32))
    # onehot = lta.LTA_func_individual_tiling(testinput)
    #testinput = tf.constant(np.array([[-15.0]]).astype(np.float32))
    #onehot = lta.LTA_func(testinput)
    onehot = lta.LTA_func(testinput)
    #print(lta.sess.run(onehot))
    # onehot = lta.LTA_func_var_add_to_input(testinput)
    onehotmat = tf.Session().run(onehot)

    import matplotlib.pyplot as plt
    for i in range(4):
        plt.plot(np.squeeze(mymat), onehotmat[:, i], label=str(i))
    plt.legend(loc='best')
    plt.show()
    print(tf.Session().run(onehot))
    # print(lta.sess.run(lta.c_list))
    #print(lta.sess.run(lta.v), lta.sess.run([tf.range(-lta.lta_bound, lta.lta_bound,
    #                                                  delta=2.*lta.lta_bound/lta.n_tiles, dtype=tf.float32)]))