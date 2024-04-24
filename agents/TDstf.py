import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
from agents.baselearner import baselearner
from utils.logger import logger
import random
from utils.utils import backfromlinear, convert2linear, introduce_noisy_labels,\
    get_cond_noise, sample_tp, getP, compute_exptytp
from networks.networks import create_linear_regression_nn, create_poisson_regression_linear
from networks.networks import create_regression_nn, create_softmax_classification_nn, create_poisson_nn,\
    update_target_nn_move, update_target_nn_assign
from networks.imnetworks import create_linear_classification, create_resnet20, create_minicnn, create_cnn


class tdlineartf(baselearner):
    def __init__(self, params):
        super(tdlineartf, self).__init__(params)
        tf.set_random_seed(self.seed)

        print("TDtf algorithm is running !!!!!!!!!!! ", self.config.dim, self.config.lam, self.config.gamma,
              self.config.learning_rate)
        self.sess = None
        ''' define NN models '''
        if 'LinearPoi' in self.name:
            self.input, self.target, self.logit, self.prediction, self.loss, _, _ \
                = create_poisson_regression_linear(self.name, self.config)
        elif 'LinearReg' in self.name:
            self.input, self.target, self.logit, self.prediction, self.loss, _, _ \
                = create_linear_regression_nn(self.name, self.config)
        elif 'LinearCla' in self.name:
            self.input, self.target, self.logit, self.prediction, self.loss, _ \
                = create_linear_classification(self.name, self.config)

        if self.config.Optimizer == 'Adam':
            self.params_update = tf.train.AdamOptimizer(self.config.learning_rate, self.config.beta1, self.config.beta2).minimize(self.loss)
        else:
            self.params_update = tf.train.GradientDescentOptimizer(self.config.learning_rate).minimize(self.loss)

        self.sess = tf.Session() if self.sess is None else self.sess
        self.sess.run(tf.global_variables_initializer())
        self.loginfo = logger()

        self.empirical_mean = 0.
        self.t = random.sample(range(self.mydatabag.x_train.shape[0]), k=self.config.mini_batch_size)
        self.tp = None
        self.curnoise = 0.
        self.nextnoise = None
        self.expt_ytp = None
        self.n = 0
        ''' if P is none, uniform const will be used by default '''
        if 'TD' in self.name:
            self.P = getP(self.config.Ptype, self.mydatabag.name, self.mydatabag.x_train, self.mydatabag.y_train, self.config.epsilon)
            if self.config.expt_ytp > 0:
                self.expt_ytp = compute_exptytp(self.P, convert2linear(self.mydatabag.name, self.mydatabag.y_train))
        elif 'WP' in self.name:
            self.P = getP(self.config.Ptype, self.mydatabag.name, self.mydatabag.x_train, self.mydatabag.y_train, self.config.epsilon)
            self.train = self.train_wp
        elif 'TD' not in self.name:
            ''' this reduces to the standard regression '''
            self.train = self.train_wop

    def get_ytd(self, reward, x_tp):
        predxtp = self.sess.run(self.logit, feed_dict={self.input: x_tp})
        tdtarget = reward.reshape(predxtp.shape) + self.config.gamma * predxtp
        tdtarget = backfromlinear(self.mydatabag.name, tdtarget)
        return  tdtarget

    def train(self, X, Y):
        self.tp = sample_tp(self.mydatabag, self.config, self.P, self.t)

        #self.nextnoise \
        #    = get_cond_noise(self.mydatabag.name, self.curnoise, self.config.noiselevel,
        #                     self.config.mini_batch_size, self.config.noise_rho)

        x_t = self.mydatabag.x_train[self.t, :]
        y_t = self.mydatabag.y_train[self.t] #+ self.curnoise

        x_tp = self.mydatabag.x_train[self.tp, :]
        y_tp = self.mydatabag.y_train[self.tp] #+ self.nextnoise

        # compute reward
        reward = compute_reward(self.mydatabag, self.P, self.expt_ytp, y_t, y_tp, self.tp, self.config.gamma)

        # compute td target
        ytd = self.get_ytd(reward, x_tp)

        ''' updating rule here '''
        self.sess.run(self.params_update, feed_dict={self.input: x_t, self.target: ytd})

        self.t = self.tp
        #self.curnoise = self.nextnoise
        self.n += 1

    def train_wp(self, X, Y):
        self.tp = sample_tp(self.mydatabag, self.config, self.P, self.t)

        #self.nextnoise \
        #    = get_cond_noise(self.mydatabag.name, self.curnoise, self.config.noiselevel,
        #                     self.config.mini_batch_size, self.config.noise_rho)

        x_t = self.mydatabag.x_train[self.t, :]
        y_t = self.mydatabag.y_train[self.t] #+ self.curnoise

        ''' updating rule here '''
        self.sess.run(self.params_update, feed_dict={self.input: x_t, self.target: y_t})

        self.t = self.tp
        #self.curnoise = self.nextnoise
        self.n += 1

    def train_wop(self, X, Y):
        #self.nextnoise \
        #    = get_cond_noise(self.mydatabag.name, self.curnoise, self.config.noiselevel,
        #                     self.config.mini_batch_size, self.config.noise_rho)

        ''' updating rule here '''
        self.sess.run(self.params_update, feed_dict={self.input: X, self.target: Y})
        #self.curnoise = self.nextnoise
        self.n += 1

    def predict(self, X):
        if len(X.shape) < 2:
            X = X.reshape((-1, self.config.dim))
        predictions = self.sess.run(self.prediction, feed_dict={self.input: X})
        return np.squeeze(predictions)

from utils.utils import get_cor_noise
class tdnns(baselearner):
    def __init__(self, params):
        super(tdnns, self).__init__(params)
        tf.set_random_seed(self.seed)

        print("TDnns algorithm is running !!!!!!!!!!! ", self.config.dim, self.config.lam, self.config.gamma,
              self.config.learning_rate)
        # print("imbalanced ratio: ", np.sum(self.mydatabag.y_test,axis=0)/self.mydatabag.y_test.shape[0])
        self.sess = None
        self.is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
        self.tgis_training = tf.placeholder(tf.bool, shape=(), name='is_training')
        ''' for simple set nh1=nh2 '''
        self.config.n_h2 = self.config.n_h1
        create_nn_func = None
        ''' define NN models '''
        if 'PoissonNN' in self.name:
            self.input, self.target, self.logit, self.prediction, self.loss, _, self.tvars \
                = create_poisson_nn(self.name, self.config)
            create_nn_func = create_poisson_nn
        elif 'RegNN' in self.name:
            self.input, self.target, self.logit, self.prediction, self.loss, _, self.tvars \
                = create_regression_nn(self.name, self.config)
            create_nn_func = create_regression_nn
        elif 'ClassifyFcNN' in self.name:
            self.input, self.target, self.logit, self.prediction, self.loss, _, self.tvars \
                = create_softmax_classification_nn(self.name, self.config)
            create_nn_func = create_softmax_classification_nn
        elif 'CNN' in self.name or 'Res20' in self.name:
            # Detect if a GPU is available
            gpu_available = tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)

            if gpu_available:
                print("GPU is available. Using GPU for computation.")
                device = '/gpu:0'
            else:
                print("No GPU available. Using CPU for computation.")
                device = '/cpu:0'

            # Create a TensorFlow session
            session_config = tf.ConfigProto(allow_soft_placement=True)
            session_config.gpu_options.allow_growth = True  # Allocate GPU memory as needed
            self.sess = tf.Session(config=session_config)

            with tf.device(device):
                if 'MiniCNN' in self.name:
                    self.input, self.target, self.logit, self.prediction, self.loss, self.tvars \
                        = create_minicnn(self.name, self.config)
                    self.tginput, _, self.tglogit, _, _, self.tgtvars = create_minicnn(self.name+'tg', self.config)
                elif 'CNN' in self.name:
                    self.input, self.target, self.logit, self.prediction, self.loss, self.tvars \
                        = create_cnn(self.name, self.config)
                    self.tginput, _, self.tglogit, _, _, self.tgtvars = create_cnn(self.name+'tg', self.config)
                elif 'Res20' in self.name:
                    self.input, self.target, self.is_training, self.logit, self.prediction, self.loss, self.tvars \
                        = create_resnet20(self.name, self.config)
                    self.tginput, _, self.tgis_training, self.tglogit, _, _, self.tgtvars = create_resnet20(self.name + 'tg', self.config)

        if create_nn_func is not None:
            self.tginput, _, self.tglogit, _, _, _, self.tgtvars \
                = create_nn_func(self.name + 'tg', self.config)

        if self.config.Optimizer == 'Adam':
            self.params_update = tf.train.AdamOptimizer(self.config.learning_rate, self.config.beta1, self.config.beta2).minimize(self.loss)
        else:
            self.params_update = tf.train.GradientDescentOptimizer(self.config.learning_rate).minimize(self.loss)

        ''' define target nn and updating rule for target nn '''
        self.target_params_init = update_target_nn_assign(self.tgtvars, self.tvars)
        self.target_params_update = update_target_nn_move(self.tgtvars, self.tvars, self.config.tau)

        self.sess = tf.Session() if self.sess is None else self.sess
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.target_params_init)

        self.loginfo = logger()

        self.t = random.sample(range(self.mydatabag.x_train.shape[0]), k=self.config.mini_batch_size)
        self.tp = None
        self.curnoise = 0.
        self.nextnoise = None
        self.expt_ytp = None
        self.n = 0
        ''' if P is none, uniform const will be used by default '''
        if 'TD' in self.name:
            self.P = getP(self.config.Ptype, self.mydatabag.name, self.mydatabag.x_train, self.mydatabag.y_train, self.config.epsilon)
            if self.config.expt_ytp > 0:
                self.expt_ytp = compute_exptytp(self.P, convert2linear(self.mydatabag.name, self.mydatabag.y_train))
            if self.config.woconv == 1:
                self.train = self.train_woconv
        elif 'WP' in self.name:
            self.P = getP(self.config.Ptype, self.mydatabag.name, self.mydatabag.x_train, self.mydatabag.y_train, self.config.epsilon)
            self.train = self.train_wp
        elif 'TD' not in self.name:
            ''' this reduces to the standard regression '''
            self.train = self.train_wop
        ''' add correlated noise to training target '''
        if self.config.noiselevel > 0:
            noise = self.config.noiselevel*get_cor_noise(self.mydatabag.y_train.shape[0], 0.95)
            # noise = np.random.normal(0., 50., self.mydatabag.y_train.shape)
            self.mydatabag.y_train += noise
            print(noise[:10])
            # self.mydatabag.y_train += np.random.normal(0., 10., self.mydatabag.y_train.shape)
            # self.P = cov

    def get_ytd(self, reward, x_tp):
        predxtp = self.sess.run(self.tglogit, feed_dict={self.tginput: x_tp, self.tgis_training: False})
        tdtarget = reward.reshape(predxtp.shape) + self.config.gamma * predxtp
        tdtarget = backfromlinear(self.mydatabag.name, tdtarget)
        return  np.squeeze(tdtarget)    

    def train(self, X, Y):
        self.tp = sample_tp(self.mydatabag, self.config, self.P, self.t)

        #self.nextnoise \
        #    = get_cond_noise(self.mydatabag.name, self.curnoise, self.config.noiselevel,
        #                     self.config.mini_batch_size, self.config.noise_rho)

        x_t = self.mydatabag.x_train[self.t, :]
        y_t = self.mydatabag.y_train[self.t] #+ self.curnoise

        x_tp = self.mydatabag.x_train[self.tp, :]
        y_tp = self.mydatabag.y_train[self.tp] #+ self.nextnoise

        # compute reward
        reward = compute_reward(self.mydatabag, self.P, self.expt_ytp, y_t, y_tp, self.tp, self.config.gamma)

        # compute td target
        ytd = self.get_ytd(reward, x_tp)

        ''' updating rule here '''
        self.sess.run(self.params_update, feed_dict={self.input: x_t, self.target: ytd, self.is_training: True})
        self.sess.run(self.target_params_update)

        self.t = self.tp
        # self.curnoise = self.nextnoise
        self.n += 1

    def train_woconv(self, X, Y):
        self.tp = sample_tp(self.mydatabag, self.config, self.P, self.t)

        x_t = self.mydatabag.x_train[self.t, :]
        y_t = self.mydatabag.y_train[self.t] #+ self.curnoise

        x_tp = self.mydatabag.x_train[self.tp, :]
        y_tp = self.mydatabag.y_train[self.tp] #+ self.nextnoise

        # compute reward
        reward = y_t - self.config.gamma * y_tp

        # compute td target
        predxtp = self.sess.run(self.tglogit, feed_dict={self.tginput: x_tp, self.tgis_training: False})
        predxtp = np.eye(predxtp.shape[1])[np.argmax(predxtp, axis=1)] if predxtp.shape[1] > 1 else np.squeeze(predxtp)
        ytd = reward.reshape(predxtp.shape) + self.config.gamma * predxtp

        ''' updating rule here '''
        self.sess.run(self.params_update, feed_dict={self.input: x_t, self.target: ytd, self.is_training: True})
        self.sess.run(self.target_params_update)

        self.t = self.tp
        self.n += 1

    def train_wp(self, X, Y):
        self.tp = sample_tp(self.mydatabag, self.config, self.P, self.t)

        #self.nextnoise \
        #    = get_cond_noise(self.mydatabag.name, self.curnoise, self.config.noiselevel,
        #                     self.config.mini_batch_size, self.config.noise_rho)

        x_t = self.mydatabag.x_train[self.t, :]
        y_t = self.mydatabag.y_train[self.t] #+ self.curnoise

        ''' updating rule here '''
        self.sess.run(self.params_update, feed_dict={self.input: x_t, self.target: y_t, self.is_training: True})

        self.t = self.tp
        #self.curnoise = self.nextnoise
        self.n += 1

    def train_wop(self, X, Y):
        #self.nextnoise \
        #   = get_cond_noise(self.mydatabag.name, self.curnoise, self.config.noiselevel,
        #                     self.config.mini_batch_size, self.config.noise_rho)

        ''' updating rule here '''
        self.sess.run(self.params_update, feed_dict={self.input: X, self.target: Y, self.is_training: True})
        #self.curnoise = self.nextnoise
        self.n += 1

    def predict(self, X):
        if len(X.shape) < 2:
            X = X.reshape((-1, self.config.dim))
        predictions = self.sess.run(self.prediction, feed_dict={self.input: X, self.is_training: True})
        return np.squeeze(predictions)


# compute reward
def compute_reward(mydatabag, P, expt_ytp, y_t, y_tp, tp, gamma):
    if expt_ytp is None:
        reward = convert2linear(mydatabag.name, y_t) - gamma * convert2linear(mydatabag.name, y_tp)
    else:
        if isinstance(expt_ytp, float) or isinstance(P, float):
            eytp = expt_ytp
        elif expt_ytp.shape[0] == mydatabag.x_train.shape[0]:
            eytp = expt_ytp[tp]
        elif mydatabag.y_train.shape[1] == P.shape[0]:
            eytp = expt_ytp[np.argmax(y_tp, axis=1)]
        reward = convert2linear(mydatabag.name, y_t) - gamma * eytp # already converted to real
    return reward