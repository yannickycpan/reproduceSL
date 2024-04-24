import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import random
from agents.baselearner import baselearner
from utils.utils import add_noise
from networks.imnetworks import create_linear_classification

class image_linearclassifier(baselearner):
    def __init__(self, params):
        super(image_linearclassifier, self).__init__(params)
        self.n = 0
        self.input, self.target, _, self.prediction, self.loss, self.tvars\
                = create_linear_classification(self.name, self.config)

        self.x_train = self.mydatabag.x_train
        self.y_train = self.mydatabag.y_train
        # self.y_train = add_noise(self.config.n_classes, self.mydatabag.y_train, self.config.noiselevel)

        self.grad_2ndmmt = None
        self.beta_2ndmmt = 0.99

        # self.params_update = tf.keras.optimizers.SGD(self.config.learning_rate).minimize(self.loss)
        #self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(self.config.learning_rate)
        #self.grads_and_vars = self.optimizer.compute_gradients(self.loss, self.tvars)

        # self.params_update = self.secondmmt_update()
        if self.config.Optimizer == 'Adam':
            self.params_update = tf.train.AdamOptimizer(self.config.learning_rate, self.config.beta1,
                                                        self.config.beta2).minimize(self.loss)
        else:
            self.params_update = tf.train.GradientDescentOptimizer(self.config.learning_rate).minimize(self.loss)

        # self.wnorm = tf.norm(self.tvars)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def secondmmt_update(self):

        if self.grad_2ndmmt is None:
            self.grad_2ndmmt = [tf.zeros_like(grad) for (grad, var) in self.grads_and_vars]

        scaled_grad_vars = []
        for ix, (grad, var) in enumerate(self.grads_and_vars):
            # grad = grad + tf.random.normal(self.grad_2ndmmt[0].shape, mean=0.0, stddev=0.001 * 1. / (0.1*self.grad_2ndmmt[0]+1.))
            grad = grad + tf.random.normal(self.grad_2ndmmt[0].shape, mean=0.0, stddev=0.01/(self.n+1.))
            self.grad_2ndmmt[ix] = self.beta_2ndmmt * self.grad_2ndmmt[ix] \
                                   + (1.0 - self.beta_2ndmmt) * tf.square(grad)

            # grad = grad + tf.random.normal(self.grad_2ndmmt[0].shape, mean=0.0, stddev=0.00001)

            corrected_2ndmmt = self.grad_2ndmmt[ix] / (1.0 - self.beta_2ndmmt)
            scaled_grad_vars.append((grad / (tf.sqrt(corrected_2ndmmt) + 1e-7), var))
            # scaled_grad_vars.append((grad / (tf.abs(tf.sqrt(corrected_2ndmmt)-tf.reduce_mean(tf.sqrt(corrected_2ndmmt))) + 1e-7), var))
        ''' apply grad '''
        # self.grads_and_vars = [(grad / (tf.sqrt(corrected_2ndmmt) + 1e-7), var) for (grad, var) in self.grads_and_vars]
        params_update = self.optimizer.apply_gradients(scaled_grad_vars)
        return params_update

    def train(self, X, Y):
        if len(X.shape) < 2:
            X = X.reshape((-1, self.config.dim))
        # i = random.sample(range(self.x_train.shape[0]), k=self.config.mini_batch_size)
        self.sess.run(self.params_update, feed_dict={self.input: X, self.target: Y})
        self.n += 1.
        #if self.n % 2000 == 0:
        #    print('the solution norm is -------------------------- ', self.sess.run(self.wnorm))
        #     self.log_grad(X, Y)

    def predict(self, X):
        if len(X.shape) < 2:
            X = X.reshape((-1, self.config.dim))
        predictions = self.sess.run(self.prediction, feed_dict={self.input: X})
        #print(' linear classifier acc is ----------------------------- ',
        #      np.sum(np.argmax(self.mydatabag.y_test,axis=1)==predictions)/len(predictions))
        return np.squeeze(predictions)