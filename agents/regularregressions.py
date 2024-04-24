import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import random
from networks.networks import create_regression_nn, create_poisson_nn, \
    create_linear_regression_nn, create_poisson_regression_linear
from utils.logger import logger
from utils.utils import get_cor_noise, introduce_noisy_labels, convert2linear, get_cond_noise
from utils.tfoperations import compute_hessian
from agents.baselearner import baselearner


class linear_regression(baselearner):
    def __init__(self, params):
        super(linear_regression, self).__init__(params)
        tf.set_random_seed(self.seed)
        self.x_train = self.mydatabag.x_train
        self.y_train = self.mydatabag.y_train if self.config.n_classes is None \
            else introduce_noisy_labels(self.mydatabag.y_train, self.config.noiselevel)

        print("LR algorithm is running !!!!!!!!!!! ", self.config.dim, self.config.lam, self.config.gamma, self.config.learning_rate)

        self.x_test = self.mydatabag.x_test
        self.y_test = self.mydatabag.y_test

        if 'LinearPoisson' in self.name:
            self.input, self.target, _, self.prediction, self.loss, self.unfoldedloss, self.tvars \
                = create_poisson_regression_linear(self.name, self.config)
            print('poisson regression used!!!!!!!!!!!!!!!!!!!!!!!')
        else:
            self.input, self.target, _, self.prediction, self.loss, self.unfoldedloss, self.tvars \
                = create_linear_regression_nn(self.name, self.config)
            print('linear regression used!!!!!!!!!!!!!!!!!!!!!!!')

        # self.lrw = np.linalg.lstsq(self.x_train, self.y_train)[0]

        if self.config.Optimizer == 'Adam':
            self.params_update = tf.train.AdamOptimizer(self.config.learning_rate, self.config.beta1, self.config.beta2).minimize(self.loss)
        else:
            self.params_update = tf.train.GradientDescentOptimizer(self.config.learning_rate).minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.loginfo = logger()

        if 'Prioritized' in self.name:
            self.train = self.train_priority
            self.priorities = self.sess.run(self.unfoldedloss, feed_dict={self.input: self.mydatabag.x_train,
                                                                          self.target: self.mydatabag.y_train})
            print('prioritized linear regression used!!!!!!!!!!!!!!!!!!!!!!!')

        self.n = 0
        # self.noisearr = get_cor_noise(self.mydatabag.name, self.config.noiselevel)
        self.curnoise = np.random.normal(0., 1., self.config.mini_batch_size) * self.config.noiselevel
        self.nextnoise = None
        self.noisen = 0

    def train(self, X, Y):
        i = random.sample(range(self.x_train.shape[0]), k=self.config.mini_batch_size)

        y = self.y_train[i] + self.curnoise
        self.nextnoise \
            = get_cond_noise(self.mydatabag.name, self.curnoise, self.config.noiselevel, self.config.mini_batch_size)

        if self.config.deltatype==1:
            y = convert2linear(self.mydatabag.name, y)

        self.sess.run(self.params_update, feed_dict={self.input: self.x_train[i],
                                                     self.target: y})
        self.curnoise = self.nextnoise
        self.n += 1


    def train_priority(self, X, Y):
        inds = np.random.choice(self.mydatabag.x_train.shape[0], self.config.mini_batch_size, replace=True,
                                p=self._compute_prob(self.priorities))
        X = self.mydatabag.x_train[inds, :]
        Y = self.mydatabag.y_train[inds]
        if len(X.shape) < 2:
            X = X.reshape((-1, self.config.dim))
        if len(Y.shape) > 1:
            Y = np.squeeze(Y)
        self.sess.run(self.params_update, feed_dict={self.input: X, self.target: Y})
        if 'Full' not in self.name:
            newpriorities = self.sess.run(self.unfoldedloss, feed_dict={self.input: X,
                                                                        self.target: Y})
            self.priorities[inds] = np.squeeze(newpriorities)
        else:
            newpriorities = self.sess.run(self.unfoldedloss, feed_dict={self.input: self.mydatabag.x_train,
                                                                    self.target: self.mydatabag.y_train})
            self.priorities = np.squeeze(newpriorities)

    def predict(self, X):
        if len(X.shape) < 2:
            X = X.reshape((-1, self.config.dim))
        loss = self.sess.run(self.loss, feed_dict={self.input: self.x_test, self.target: self.y_test})
        if 'test-loss' not in self.loginfo.error_dict:
            self.loginfo.error_dict['test-loss'] = []
        self.loginfo.error_dict['test-loss'].append(loss)

        loss = self.sess.run(self.loss, feed_dict={self.input: self.x_train, self.target: self.y_train})
        if 'train-loss' not in self.loginfo.error_dict:
            self.loginfo.error_dict['train-loss'] = []
        self.loginfo.error_dict['train-loss'].append(loss)

        predictions = self.sess.run(self.prediction, feed_dict={self.input: X})
        if len(np.squeeze(predictions).shape)>1:
            predictions = np.argmax(predictions, axis=1)
        return np.squeeze(predictions)


class nnlearner(baselearner):
    def __init__(self, params):
        super(nnlearner, self).__init__(params)
        """ poisson regression is also here """
        if 'Poisson' in self.name:
            create_nn = create_poisson_nn
        else:
            create_nn = create_regression_nn

        self.input, self.target, _, self.prediction, self.loss, self.unfoldedloss, self.tvars \
                = create_nn(self.name, self.config)

        if self.config.Optimizer == 'Adam':
            self.params_update = tf.train.AdamOptimizer(self.config.learning_rate, self.config.beta1, self.config.beta2).minimize(self.loss)
        else:
            self.params_update = tf.train.GradientDescentOptimizer(self.config.learning_rate).minimize(self.loss)

        self.grad = tf.gradients(self.prediction, [self.input])[0]
        #self.grad_hidden = tf.gradients(self.prediction, [self.hidden1])[0]
        self.grad2nd = tf.gradients(self.grad, [self.input])[0]
        self.loss_grad = tf.gradients(self.loss, [self.input])[0]
        self.hessian = compute_hessian(self.prediction, self.input, self.config.dim)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.priorities = None
        if 'Prioritized' in self.name:
            self.train = self.train_priority
            self.priorities = self.sess.run(self.unfoldedloss, feed_dict={self.input: self.mydatabag.x_train,
                                                                    self.target: self.mydatabag.y_train})
            ''' the buffer only needs to maintain indexes, its size must be greater than mini-batch size '''
            if 'Buffer' in self.name:
                self.erbuffer = list(np.random.choice(self.mydatabag.x_train.shape[0],
                                                      self.config.mini_batch_size * 2, replace=False))
                self.train = self.train_buffer_priority
            print('Prioritized L2 is used ------------------------------------ ',
                  self.mydatabag.x_train.shape, self.config.learning_rate, self.config.n_h1, self.config.n_h2)
        elif self.name == 'IncrementalL2':
            self.erbuffer = list(np.random.choice(self.mydatabag.x_train.shape[0],
                                                  self.config.mini_batch_size * 2, replace=False))
            self.train = self.train_buffer_incremental
        self.n = 0

    '''
    def compute_error(self, final_log=False):
        if self.mydatabag.name == 'insurancefirsttrain':
            self.loginfo.log_error(self.predict, self.mydatabag, final_log)
            X = self.mydatabag.x_train.copy()
            X[:, -1] = 1 - X[:, -1]
            predictions = self.sess.run(self.prediction, feed_dict={self.input: X})
            # predictions = self.mydatabag.inv_transform(predictions)
            print(' predictions are :: after ', predictions)
            newdataset = np.hstack([self.mydatabag.wholedataset, predictions.reshape((-1, 1))])
            np.savetxt(fname='insurancemodal' + str(self.config.seed) + '.txt', X=newdataset, fmt='%10.7f', delimiter=',')
        else:
            self.loginfo.log_error(self.predict, self.mydatabag, final_log)
            meantarget = np.mean(self.mydatabag.y_train)
            rmse = np.sqrt(np.mean(np.square(meantarget - self.mydatabag.y_test)))
            print('the mean prediction is --------------------------------------------- ', rmse)
            if 'mean_pred' not in self.loginfo.error_dict:
                self.loginfo.error_dict['mean_pred'] = []
            self.loginfo.error_dict['mean_pred'].append(rmse)
    '''

    def train(self, X, Y):
        if len(X.shape) < 2:
            X = X.reshape((-1, self.config.dim))
        if len(Y.shape) > 1:
            Y = np.squeeze(Y)
        self.sess.run(self.params_update, feed_dict={self.input: X, self.target: Y})

    def prioritized_sampling(self):
        if self.config.subsample_ratio == 1.0:
            inds = np.random.choice(self.mydatabag.x_train.shape[0], self.config.mini_batch_size, replace=True,
                                    p=self._compute_prob(self.priorities))
            X = self.mydatabag.x_train[inds, :]
            Y = self.mydatabag.y_train[inds]
            return X, Y, inds
        subsetsize = int(self.config.subsample_ratio * self.mydatabag.x_train.shape[0])
        subsetinds = np.random.choice(np.arange(self.mydatabag.x_train.shape[0]), subsetsize, replace=False)
        inds = np.random.choice(subsetinds, self.config.mini_batch_size, replace=True,
                                p=self._compute_prob(self.priorities[subsetinds]))
        X = self.mydatabag.x_train[inds, :]
        Y = self.mydatabag.y_train[inds]
        return X, Y, inds

    def train_priority(self, X, Y):
        X, Y, inds = self.prioritized_sampling()
        if len(X.shape) < 2:
            X = X.reshape((-1, self.config.dim))
        if len(Y.shape) > 1:
            Y = np.squeeze(Y)
        self.sess.run(self.params_update, feed_dict={self.input: X, self.target: Y})
        ''' after training, update priorities '''
        if 'Full' not in self.name:
            newpriorities = self.sess.run(self.unfoldedloss, feed_dict={self.input: X, self.target: Y})
            self.priorities[inds] = np.squeeze(newpriorities)
        else:
            newpriorities = self.sess.run(self.unfoldedloss, feed_dict={self.input: self.mydatabag.x_train,
                                                                    self.target: self.mydatabag.y_train})
            self.priorities = np.squeeze(newpriorities)

    def update_erbuffer(self, sampleid):
        ''' add it to the buffer '''
        self.erbuffer.append(sampleid)
        if len(self.erbuffer) <= int(self.config.subsample_ratio * self.mydatabag.x_train.shape[0]):
            return
        ''' eliminate sample '''
        if 'Rec' in self.name:
            self.erbuffer.pop(0)
        else:
            minpriorityindex = np.argmin(self.priorities[self.erbuffer])
            self.erbuffer.pop(minpriorityindex)

    def train_buffer_priority(self, X, Y):
        sampleid = np.random.randint(0, self.mydatabag.x_train.shape[0], 1)[0]
        self.update_erbuffer(sampleid)
        inds = np.random.choice(self.erbuffer, self.config.mini_batch_size, replace=True,
                                p=self._compute_prob(self.priorities[self.erbuffer]))
        X = self.mydatabag.x_train[inds, :]
        Y = self.mydatabag.y_train[inds]
        if len(X.shape) < 2:
            X = X.reshape((-1, self.config.dim))
        if len(Y.shape) > 1:
            Y = np.squeeze(Y)
        self.sess.run(self.params_update, feed_dict={self.input: X, self.target: Y})
        ''' after training, update priorities '''
        newpriorities = self.sess.run(self.unfoldedloss, feed_dict={self.input: self.mydatabag.x_train,
                                                                    self.target: self.mydatabag.y_train})
        self.priorities = np.squeeze(newpriorities)
        self.n += 1

    def train_buffer_incremental(self, X, Y):
        sampleid = np.random.randint(0, self.mydatabag.x_train.shape[0], 1)[0]
        self.erbuffer.append(sampleid)
        if len(self.erbuffer) > int(self.config.subsample_ratio * self.mydatabag.x_train.shape[0]):
            self.erbuffer.pop(0)
        inds = np.random.choice(self.erbuffer, self.config.mini_batch_size, replace=True)
        X = self.mydatabag.x_train[inds, :]
        Y = self.mydatabag.y_train[inds]
        if len(X.shape) < 2:
            X = X.reshape((-1, self.config.dim))
        if len(Y.shape) > 1:
            Y = np.squeeze(Y)
        self.sess.run(self.params_update, feed_dict={self.input: X, self.target: Y})

    def predict(self, X):
        if len(X.shape) < 2:
            X = X.reshape((-1, self.config.dim))
        predictions = self.sess.run(self.prediction, feed_dict={self.input: X})
        return np.squeeze(predictions)

    def compute_grad_norm(self, X):
        if len(X.shape) < 2:
            X = X.reshape((-1, self.config.dim))
        grad = self.sess.run(self.grad, feed_dict={self.input: X})
        magnitude = np.linalg.norm(grad, ord=2, axis=1)
        return magnitude

    def get_representation(self, X):
        if len(X.shape) < 2:
            X = X.reshape((-1, self.config.dim))
        repres = self.sess.run(self.hidden2, feed_dict={self.input: X})
        return repres

    def compute_grad2nd_norm(self, X):
        delta = 0.0001
        if len(X.shape) < 2:
            X = X.reshape((X.shape[0], self.config.dim))
        magnitude = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            sample = X[i, :]
            grad = self.sess.run(self.grad, feed_dict={self.input: sample[None, :]})
            grad_plus = self.sess.run(self.grad, feed_dict={self.input: sample[None, :] + delta * grad})
            grad_minus = self.sess.run(self.grad, feed_dict={self.input: sample[None, :] - delta * grad})
            hess_grad = (grad_plus - grad_minus) / (2. * delta)
            norm = hess_grad.dot(hess_grad.T)
            magnitude[i] = norm[0, 0]
        return magnitude + 1e-6


class generalpowerlearner(baselearner):
    def __init__(self, params):
        super(generalpowerlearner, self).__init__(params)
        self.power = int(self.name.split('Power')[1][0])
        if 'LinearPoisson' in self.name:
            self.input, self.target, self.prediction, self.loss, self.unfoldedloss, _ \
                = create_poisson_regression_linear(self.name, self.config)
        elif 'Linear' in self.name:
            self.input, self.target, self.prediction, self.loss, self.unfoldedloss, _ \
                = create_linear_regression_nn(self.name, self.config)
        else:
            self.input, self.target, self.prediction, _, self.unfoldedloss, _, _ \
                = create_regression_nn(self.name, self.config)

        self.normalizer = tf.placeholder(tf.float32, [])
        self.finalobj = 2./3.*1.0/self.normalizer * tf.reduce_sum(tf.math.pow(self.unfoldedloss, self.power)) \
            if 'Normalized' in self.name else tf.reduce_mean(tf.math.pow(self.unfoldedloss, self.power))

        self.params_update = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.finalobj)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.n = 0

    def compute_normalizer(self, X, Y):
        unfoldedloss = self.sess.run(self.unfoldedloss, feed_dict={self.input: X, self.target: Y})
        return np.sum(unfoldedloss)

    def train(self, X, Y):
        if len(X.shape) < 2:
            X = X.reshape((-1, self.config.dim))
        if len(Y.shape) > 1:
            Y = np.squeeze(Y)
        self.sess.run(self.params_update, feed_dict={self.input: X, self.target: Y,
                                                     self.normalizer: self.compute_normalizer(X, Y)})

    def predict(self, X):
        if len(X.shape) < 2:
            X = X.reshape((-1, self.config.dim))
        predictions = self.sess.run(self.prediction, feed_dict={self.input: X})
        return np.squeeze(predictions)