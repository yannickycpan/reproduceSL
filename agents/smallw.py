import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from networks.ftaonwnetworks import create_image_classification_ftaonw
from networks.imnetworks import vectorize_vars, mask_vars, get_mask
from agents.baselearner import baselearner
from utils.utils import *


class image_classifier_smallw(baselearner):
    def __init__(self, params):
        super(image_classifier_smallw, self).__init__(params)
        self.n = 0
        self.input, self.target, self.prediction, self.loss, self.augwf, self.wall, self.tvars \
                = create_image_classification_ftaonw(self.name, self.config)

        self.thres = tf.placeholder(name='thres', dtype=tf.float32, shape=())
        self.masks = None
        self.masks_op = get_mask(self.tvars, self.thres)
        self.tvarsbackup = None

        self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
        self.params_adamupdate = self.optimizer.minimize(self.loss)
        self.initopt = tf.variables_initializer(self.optimizer.variables())

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.lastw = 0
        self.lastb = 0


    def mask_vars(self, perc, wall):
        vec = self.sess.run(self.augwf)
        thres = np.quantile(np.abs(vec), perc)
        self.masks = 1.*(np.abs(wall) > thres)
        print(' the thres is ---------------------------------- ', thres, np.sum(np.abs(vec)<thres)/vec.shape[0])
        # those masked should be zero
        self.sess.run(tf.assign(self.wall, self.masks*wall))
        return

    #def compute_error(self, final_log=False):
    #    self.loginfo.log_error_discrete(self.predict, self.mydatabag, final_log)

    def train(self, X, Y):
        self.sess.run(self.params_adamupdate, feed_dict={self.input: X, self.target: Y})
        self.n += 1
        if self.n % 500 == 0:
            augwf = self.sess.run(self.augwf)
            print(' the proportion of nonzero entry is ---------- ', np.sum(np.abs(augwf)>1e-6), np.prod(augwf.shape),
                  np.sum(np.abs(augwf)>1e-6)/np.prod(augwf.shape))
        if self.n % 1000 == 0:
            self.sess.run(self.initopt)
            if self.n % 10000 == 0:
                self.sess.run(self.wall.initializer)

    def predict(self, X):
        predictions = self.sess.run(self.prediction, feed_dict={self.input: X})
        return np.squeeze(predictions)

    def compute_error(self, final_log=False):
        print(' compute acc at step ----------------------------------------- ', self.n)
        yhat = self.predict(self.mydatabag.x_test)
        testerr = classify_error(self.mydatabag.y_test, yhat, True)
        print(' the testing acc before masking is ', 1.-testerr)
        ''' backup current params '''
        wall = self.sess.run(self.wall)
        self.tvarsbackup = wall
        for thres in [0.992, 0.99, 0.98, 0.95]:
            self.mask_vars(thres, wall)
            yhat = self.predict(self.mydatabag.x_test)
            testerr = classify_error(self.mydatabag.y_test, yhat, True)
            print(' the testing acc after masking is ', 1. - testerr)
            ''' check the sparsity of the whole NN after mask '''
            numnonzero = np.sum(np.abs(self.sess.run(self.augwf))>0)
            print(' the proportion of nonzero entires after masking is --- :: ', numnonzero/self.tvarsbackup.shape[0])
            self.sess.run(tf.assign(self.wall, self.tvarsbackup))
        return
