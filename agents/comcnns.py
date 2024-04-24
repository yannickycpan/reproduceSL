import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from agents.baselearner import baselearner
from utils.logger import logger
from utils.utils import introduce_noisy_labels, get_cond_noise
from networks.imnetworks import create_resnet20, create_minicnn


class comcnns(baselearner):
    def __init__(self, params):
        super(comcnns, self).__init__(params)
        self.sess = None

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
        tf.keras.backend.set_session(self.sess)  # Set this session as the default session

        with tf.device(device):
            if 'MiniCNN' == self.name:
                self.input, self.target, self.logit, self.prediction, self.loss, _ \
                    = create_minicnn(self.name, self.config)
            elif 'Res20' == self.name:
                self.input, self.target, self.logit, self.prediction, self.loss, _ \
                    = create_resnet20(self.name, self.config)

        if self.config.Optimizer == 'Adam':
            self.params_update = tf.train.AdamOptimizer(self.config.learning_rate, self.config.beta1, self.config.beta2).minimize(self.loss)
        else:
            self.params_update = tf.train.GradientDescentOptimizer(self.config.learning_rate).minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())
        self.loginfo = logger()

        self.empirical_mean = 0.
        self.t = 0
        self.tp = None
        self.curnoise = None
        self.nextnoise = None
        self.n = 0

    def train(self, X, Y):
        if self.curnoise is None:
            self.curnoise = np.random.normal(0., 1., self.config.mini_batch_size) * self.config.noiselevel \
                if self.config.noiselevel > 0 else 0.0

        self.nextnoise \
            = get_cond_noise(self.mydatabag.name, self.curnoise, self.config.noiselevel, self.config.mini_batch_size)

        ''' updating rule here '''
        self.sess.run(self.params_update, feed_dict={self.input: X, self.target: Y + self.curnoise})

        self.t = self.tp
        self.curnoise = self.nextnoise
        self.n += 1

    ''' for predicting integers '''

    def predict(self, X):
        if len(X.shape) < 2:
            X = X.reshape((-1, self.config.dim))
        predictions = self.sess.run(self.prediction, feed_dict={self.input: X})
        return np.squeeze(predictions)