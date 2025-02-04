import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
# python data types
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.python.client import timeline
from tensorflow.python.ops import clip_ops
from optimisers.adamax import AdamaxOptimizer
from tqdm import tqdm

DTYPE = tf.float32
NP_DTYPE = np.float32

tfd = tf.contrib.distributions
tfb = tfd.bijectors

np.random.seed(1)
tf.set_random_seed(1)

def softminus(x):
    return tf.log(tf.exp(x) - 1)

def softplus_(x):
    return tf.log(1 + tf.exp(x))


class init_dist(tfd.Normal):

    def __init__(self, loc, scale, batch_dims, target_dims):
        self.batch_dims = batch_dims
        self.target_dims = target_dims
        tfd.Normal.__init__(self, loc=loc, scale=scale)

    def slp(self, p_val):
        sample = self.sample(p_val)
        log_prob = tf.reduce_sum(self.log_prob(
            sample)[:, -self.batch_dims:], axis=1)
        return sample, log_prob


class Bivariate_Normal():
    '''
    bivariate batched normal dist
    '''

    def __init__(self, mu, chol):
        self.mu = tf.expand_dims(mu, 2)
        # self.mu = mu
        self.chol = chol
        self.det = tf.reduce_prod(tf.matrix_diag_part(chol), axis=1) ** 2
        self.cov_inv = tf.matrix_inverse(chol @ tf.transpose(chol, [0, 2, 1]))
        # if bijector_chain is not None:
        #     self.dist = tfd.TransformedDistribution(
        #         distribution=tfd.MultivariateNormalTriL(loc=self.mu, scale_tril=self.chol),
        #         bijector=bijector_chain)

    def normal_log_prob(self, x):
        return (- (1 / 2) * tf.log(self.det) 
                + tf.squeeze(-0.5 * tf.transpose((x - self.mu), [0, 2, 1])
                             @ self.cov_inv @ (x - self.mu))
                - np.log(2 * np.pi))
        
    def log_prob(self, x, bijector_chain=None):
        '''
        log prob of bivariate normal dist with bijector chain
        '''
        if bijector_chain is None:
            return self.normal_log_prob(tf.expand_dims(x, 2))      
        else:
            # return self.dist.log_prob(x)
            y = bijector_chain.inverse(x)
            log_prob = self.normal_log_prob(tf.expand_dims(y, 2))
            log_det_jacobian = bijector_chain.inverse_log_det_jacobian(x)
            return log_prob + log_det_jacobian


class IAF():

    """
    single-stack local IAF with feature injection
    """

    def __init__(self, network_dims, theta, ts_feats, feat_dims = 10):
        self.network_dims = network_dims
        self.num_layers = len(network_dims)
        self.theta = theta
        self.ts_feats = ts_feats
        self.feat_dims = feat_dims

    def _create_flow(self, base_dist, p_val, kernel_len, batch_dims, target_dims, first_flow = False):
        base_sample, self.base_logprob = base_dist.slp(p_val)

        feat_layers = [self.ts_feats[:, :-1, :]]
        for i in range(3):
            feat_layers.append(tf.layers.dense(
                inputs=feat_layers[-1], units=self.network_dims[0], activation=tf.nn.elu))
        feat_layers.append(tf.transpose(tf.layers.dense(
            inputs=feat_layers[-1], units=self.feat_dims, activation=tf.nn.elu), [0, 2, 1]))

        convnet_inp = tf.concat(
            [tf.expand_dims(base_sample[:, :-1], 2), feat_layers[-1]], axis=2)

        layer1A = tf.layers.conv1d(inputs=convnet_inp, filters=network_dims[0],
                                   kernel_size=kernel_len, strides=1, padding='valid', activation=None)
        layer1B1 = tf.layers.dense(
            inputs=self.theta, units=self.network_dims[0], activation=None)
        layer1B2 = tf.layers.dense(
            inputs=layer1B1, units=self.network_dims[0], activation=None)
        layer1B = tf.layers.dense(
            inputs=layer1B2, units=self.network_dims[0], activation=None)

        layer1C = layer1A + tf.expand_dims(layer1B, 1)
        layers = [tf.nn.elu(layer1C)]

        for i in range(1, self.num_layers - 1):
            layers.append(tf.layers.conv1d(
                inputs=layers[-1], filters=self.network_dims[i], kernel_size=1, strides=1, activation=tf.nn.elu))
            layers.append(tf.layers.batch_normalization(layers[-1]))
        layers.append(tf.layers.conv1d(inputs=layers[-1], filters=2, kernel_size=1, strides=2, activation=None))

        mu_temp, sigma_temp = tf.split(layers[-1], 2, axis=2)
        mu_aug = tf.reshape(tf.concat([tf.zeros(tf.shape(mu_temp)), mu_temp], axis = 2), [p_val, -1])
        sigma_aug = tf.reshape(tf.concat([tf.ones(tf.shape(sigma_temp)), tf.nn.softplus(sigma_temp) + 1e-10], axis = 2), [p_val, -1])

        self.sigma_log = tf.log(sigma_aug[:, -batch_dims:])
        self.output = base_sample[:, kernel_len:] * sigma_aug + mu_aug

    def slp(self, *args):
        logprob = self.base_logprob - tf.reduce_sum(self.sigma_log, axis=1)
        return self.output, logprob


class Flow_Stack():

    """
    Create locally variant IAF stack
    """

    def __init__(self, flows, kernel_len, batch_dims, target_dims):
        base_dims = kernel_len * no_flows + batch_dims + 2
        base_dist = init_dist(loc=[0.0] * base_dims, scale=[1e0] *
                              base_dims, batch_dims=batch_dims, target_dims=target_dims)
        flows.insert(0, base_dist)

        for i in range(1, len(flows)):
            if i == 1:
                flows[i]._create_flow(
                    flows[i - 1], p_val, kernel_len, batch_dims, target_dims, first_flow = True)
            else:
                flows[i]._create_flow(
                    flows[i - 1], p_val, kernel_len, batch_dims, target_dims)

        self.output = flows[-1]

    def slp(self):
        return self.output.slp()


class Permute():
    '''
    class to permute IAF
    '''

    def __init__(self, permute_tensor):
        '''
        :params permute_index: permutations as list
        '''
        self.permute_tensor = permute_tensor

    def _create_flow(self, base_dist, *args):
        '''
        function to permute base dist order
        :params base_dist: base dist to permute
        '''

        sample, self.log_prob = base_dist.slp()
        shape = tf.shape(sample)
        self.sample = tf.scatter_nd(self.permute_tensor, sample, shape)

    def slp(self, *args):
        return self.sample, self.log_prob


class VI_SSM():

    def __init__(self, obs, obs_bin, time_till, x0_mean, x0_std, theta_dist, priors, dt, T, p_val, kernel_len, batch_dims, network_dims, target_dims, no_flows, feat_window, learn_rate = 1e-3, pre_train=True):
        # raw inputs -> class variables
        self.obs = obs
        self.obs_bin = obs_bin
        obs_flatten = np.reshape(obs, -1, 'F')
        self.flow_dims = 2
        self.theta_dist = theta_dist
        self.theta = theta_dist.sample(p_val)
        self.theta_log_prob = theta_dist.log_prob(self.theta)
        self.priors = priors
        self.dt = dt
        self.p_val = p_val
        self.kernel_len = kernel_len
        self.batch_dims = batch_dims
        self.network_dims = network_dims
        self.target_dims = target_dims
        self.no_flows = no_flows
        self.theta_eval = self._theta_strech()
        self.diffusivity = tf.placeholder(DTYPE, 1)
        self.pre_train = pre_train
        self.learn_rate = learn_rate
        self.kernel_ext = self.kernel_len * self.no_flows + \
            self.flow_dims * self.batch_dims + 2

        # augementing raw inputs
        self.obs_pad_store = []
        for i in range(0, feat_window*5, 5):
            self.obs_pad_store.append(np.concatenate(
                (np.zeros(self.no_flows * self.kernel_len + self.flow_dims - i), obs_flatten, np.zeros(i)), axis=0))
        self.time_pad = np.concatenate((np.zeros(self.no_flows * self.kernel_len + self.flow_dims),
                                        np.repeat(np.arange(0, T + dt, dt), self.flow_dims)), axis=0)
        print('time_pad:', self.time_pad.shape)
        time_till_pad = np.reshape(np.repeat(np.arange(np.round((self.no_flows * self.kernel_len + self.flow_dims) * (
            self.dt / self.flow_dims), 1), 0., -self.dt), self.flow_dims), (self.flow_dims, -1), 'F')
        self.time_till = np.reshape(np.concatenate(
            (time_till_pad, time_till), 1), -1, 'F')
        self.bin_feats = np.float32(np.concatenate(
            (np.zeros(self.no_flows * self.kernel_len + self.flow_dims), np.ones(self.target_dims * self.flow_dims)), axis=0))
        self.mask_vals = np.concatenate(
            (np.zeros((2, 1)), np.ones((self.flow_dims, self.target_dims))), axis=1)
        self.shift_vals = np.concatenate(
            (np.expand_dims(x0_mean, 1), np.zeros((self.flow_dims, self.target_dims))), axis=1)
        self.x0_mean = x0_mean
        self.x0_std = x0_std
        
        
        # perm index
        perm_list = []
        for j in range(self.p_val):
            for i in range(1, self.kernel_ext - self.kernel_len, 2):
                perm_list.append([j, i])
                perm_list.append([j, i - 1])
        self.perm_index = tf.constant(np.reshape(
            np.array(perm_list), (self.p_val, -1, 2)), tf.int32)

        # model placeholders
        self.time_feats = tf.placeholder(
            shape=[self.p_val, self.kernel_len * self.no_flows + self.flow_dims * self.batch_dims + self.flow_dims, feat_window + 3], dtype=DTYPE)
        self.obs_eval = tf.transpose(tf.reshape(
            self.time_feats[:, -self.flow_dims * self.batch_dims:, 0], [self.p_val, -1, 2]), [0, 2, 1]) #get only obs
        self.mask = tf.placeholder(
            shape=[self.p_val, self.flow_dims, self.batch_dims + 1], dtype=DTYPE)
        self.shift = tf.placeholder(
            shape=[self.p_val, self.flow_dims, self.batch_dims + 1], dtype=DTYPE)
        self.bin_feed = tf.placeholder(
            shape=[self.p_val, self.flow_dims, self.batch_dims], dtype=DTYPE)
        
    def transform_dist(self, base_dist):
        transforms = tfb.Chain([
            tfb.AffineScalar(shift=1.0, scale=1.0),  
            tfb.Softplus(),                 
            tfb.AffineScalar(shift=-1.0, scale=1.0)    
        ])
        return tfd.TransformedDistribution(
            distribution=base_dist, bijector=transforms)
        
    def _theta_strech(self):
        slice_stash = []
        for i in range(len(self.priors)):
            slice_stash.append(tf.reshape(tf.tile(tf.expand_dims(
                (self.theta[:, i]), 1), [1, self.batch_dims-1]), [-1]))
        return slice_stash

    def _ELBO(self):
        self.y_loc = self.lf_sample[:, :, 1:]
        self.y_scale = self.theta[:,-1] * self.lf_sample[:, :, 1:]
        y_dist = self.transform_dist(tfd.Normal(loc=self.y_loc, scale=self.y_scale))
        y_prob = y_dist.log_prob(self.obs_eval)
        obs_log_prob = tf.reduce_sum(tf.reshape((y_prob * self.bin_feed), [self.p_val, -1]), 1)

        def alpha(x1, x2, theta):
            drift_vec = tf.concat([tf.reshape(theta[0] * x1 - theta[1] * x1 * x2, [-1, 1]),
                                   tf.reshape(theta[1] * x1 * x2 - theta[2] * x2, [-1, 1])], axis=1)
            return drift_vec

        def sqrt_beta(x1, x2, theta):
            a = tf.reshape(
                tf.sqrt(theta[0] * x1 + theta[1] * x1 * x2), [-1, 1, 1])
            b = tf.reshape(- theta[1] * x1 * x2, [-1, 1, 1]) / a
            c = tf.sqrt(tf.reshape(
                theta[1] * x1 * x2 + theta[2] * x2, [-1, 1, 1]) - b ** 2)
            zeros = tf.zeros(tf.shape(a))
            beta_chol = tf.concat(
                [tf.concat([a, zeros], 2), tf.concat([b, c], 2)], 1)
            return beta_chol
        
        #p(x)
        flow_head = self.lf_sample[:, :, 1:-1]
        self.lf_sample_flat = tf.concat(
            [tf.reshape(self.lf_sample[:, 0, 2:], [-1, 1]),
            tf.reshape(self.lf_sample[:, 1, 2:], [-1, 1])], 1) 
        self.xt_plus_dt = tf.concat([tf.reshape(flow_head[:, 0, :], [-1, 1]),
                                    tf.reshape(flow_head[:, 1, :], [-1, 1])], 1)
        self.sde_drift = self.dt * alpha(tf.reshape(flow_head[:, 0, :], [-1]), 
                                         tf.reshape(flow_head[:, 1, :], [-1]), 
                                         self.theta_eval)
        self.sde_mu = self.sde_drift + self.xt_plus_dt
        self.sde_chol = tf.sqrt(self.dt) * sqrt_beta(
            tf.reshape(flow_head[:, 0, :], [-1]), 
            tf.reshape(flow_head[:, 1, :], [-1]), 
            self.theta_eval)
        # sde_dist = tfd.TransformedDistribution(
        # distribution=tfd.MultivariateNormalTriL(
        #     loc=self.sde_mu,
        #     scale_tril=self.sde_chol),
        # bijector=tfb.Chain([
        #     tfb.Affine(shift=[1.,1.], scale_diag=[1.,1.]),  
        #     tfb.Softplus(event_ndims=2),                      
        #     tfb.Affine(shift=[-1.,-1.], scale_diag=[1.,1.])])
        # )
        # sde_dist = tfd.MultivariateNormalTriL(loc=self.sde_mu, scale_tril=self.sde_chol)
        # self.sde_neg_sample = tfd.MultivariateNormalTriL(loc=self.sde_mu,scale_tril=self.sde_chol).sample()
        # self.sde_sample = sde_dist.sample()
        # SDE_log_prob = tf.reduce_sum(tf.reshape(
        #     sde_dist.log_prob(self.lf_sample_flat), [self.p_val, -1]), 1)
        
        SDE_log_prob = tf.reduce_sum(tf.reshape(
            Bivariate_Normal(mu=self.sde_mu,chol=self.sde_chol).log_prob(
                self.lf_sample_flat, bijector_chain=None), 
            [self.p_val, -1]), 1) 
        
        ## p(x0)
        x0_sample = self.lf_sample[:, :, 1] # (50, 2)
        x0_transforms = tfb.Chain([
            tfb.Affine(shift=[1.,1.], scale_diag=[1.,1.]),  
            tfb.Softplus(event_ndims=2),                      
            tfb.Affine(shift=[-1.,-1.], scale_diag=[1.,1.])    
                ])
        x0_dist_init = tfd.MultivariateNormalDiag(loc=self.x0_mean, scale_diag=self.x0_std)
        x0_dist = tfd.TransformedDistribution(distribution=x0_dist_init, bijector=x0_transforms)
        self.x0_log_prob = x0_dist.log_prob(x0_sample)
        SDE_log_prob += self.x0_log_prob
        
        prior_mean = [item[0] for item in self.priors]
        prior_scale = [item[1] for item in self.priors]

        prior_log_prob = tfd.TransformedDistribution(
            tfd.MultivariateNormalDiag(
                loc=prior_mean, scale_diag=prior_scale), bijector=tfb.Softplus(event_ndims=2)
            ).log_prob(self.theta)
        
        #log p(x) - log q(x) + log p(y|x) + log p(theta) - log q(theta)
        ELBO = (self.target_dims / self.batch_dims) * (
            SDE_log_prob - self.lf_log_prob + obs_log_prob
            ) + prior_log_prob - self.theta_log_prob

        return ELBO, SDE_log_prob, obs_log_prob, prior_log_prob

    def build_flow(self):
        flows = []
        for i in range(self.no_flows):
            flows.append(IAF(network_dims=self.network_dims, theta=self.theta,
                             ts_feats=self.time_feats, feat_dims = self.kernel_ext - 1 - i * kernel_len))
            if i == 0:
                flows.append(Permute(permute_tensor=self.perm_index))
            else:
                flows.append(
                    Permute(permute_tensor=self.perm_index[:, :-(i * self.kernel_len), :]))
        ## SSM gives log sigma (sigma is already softplused in IAF, this is log of softplused sigma) and x=mu+sigma*z
        #stack permutations and IAF
        self.SSM = Flow_Stack(flows[:-1], self.kernel_len,
                         self.batch_dims * self.flow_dims, self.target_dims)
        #lf_sample_init: x samples output of q(x|theta) with IAF (50, 102)
        #lf_log_prob_init: log q(x) (50,)
        lf_sample_init, lf_log_prob_init = self.SSM.slp() #slp: sample, log probability
        self.lf_sample_neg = tf.transpose(tf.reshape(
            lf_sample_init, [self.p_val, -1, 2]), [0, 2, 1]) #lf_sample_neg: (p_val=50, (u-v)=2, M+1=51)
        # transform
        qx_transform = tfb.Chain([
            tfb.Affine(shift=[1.,1.], scale_diag=[1.,1.]),  
            tfb.Softplus(event_ndims=2)                   
            # tfb.Affine(shift=[-1.,-1.], scale_diag=[1.,1.])    
        ])
        
        def trp(x):
            return tf.transpose(x, [0, 2, 1])
        #self.lf_sample (50, 2, 51): 50 batches of
        # [[u0, softplus(u1_q), ..., softplus(uM_q)], 
        #  [v0, softplus(v1_q), ..., softplus(vM_q)]]
        lf_sample_neg = trp(self.lf_sample_neg) # (50, 51, 2)
        #modifying log q(x) to accomodate softplus
        self.lf_sample = trp(qx_transform.forward(lf_sample_neg)) * self.mask + self.shift # (50, 2, 51)
        # self.lf_sample = trp(qx_transform.forward(lf_sample_neg))
        
        self.lf_log_prob = lf_log_prob_init + qx_transform.inverse_log_det_jacobian(
            trp(self.lf_sample[:, :, 1:])) #log q(x) from normalizing flow # (50,)       
        loss, self.sde_loss, self.obs_loss, prior_prob = self._ELBO()
        self.mean_loss = tf.reduce_mean(loss)
        
        self.t1 = AdamaxOptimizer(
            learning_rate=1e-3, beta1=0.9).minimize((self.lf_sample - 75) ** 2)

        self.t2 = AdamaxOptimizer(learning_rate=1e-3, beta1=0.9).minimize(
            (self.theta - softplus_(tf.tile([[-1.0, -6.0, -1.0, -2.0]], [self.p_val, 1]))) ** 2)

        # do something nicer with this!
        theta_pos_index = [True, True, True]
        with tf.name_scope('loss'):
            tf.summary.scalar('ELBO', self.mean_loss)
            tf.summary.scalar(
                'SDE_log_prob p(x)', (self.target_dims / self.batch_dims) * tf.reduce_mean(self.sde_loss))
            tf.summary.scalar('theta_log_prob q(theta)',
                              tf.reduce_mean(self.theta_log_prob))
            tf.summary.scalar(
                'obs_log_prob p(y|x)', (self.target_dims / self.batch_dims) * tf.reduce_mean(self.obs_loss))
            tf.summary.scalar(
                'path_log_prob q(x)', (self.target_dims / self.batch_dims) * tf.reduce_mean(self.lf_log_prob))
            tf.summary.scalar('truth_log_prob q(theta*)', -self.theta_dist.log_prob(
                softplus_([[-1.0, -6.0, -1.0, -2.0]]))[0])
            # theta summaries

            for i in range(len(theta_pos_index)):
                if theta_pos_index[i]:
                    tf.summary.histogram(str(i),
                        self.theta[:, i], family='parameters')
                else:
                    tf.summary.histogram(
                        str(i), self.theta[:, i], family='parameters')

        with tf.name_scope('optimize'):
            opt = AdamaxOptimizer(learning_rate=self.learn_rate, beta1=0.95)
            gradients, variables = zip(
                *opt.compute_gradients(-loss))
            global_norm = tf.global_norm(gradients)
            self.gradients, _ = tf.clip_by_global_norm(gradients, 1e9)
            self.train_step = opt.apply_gradients(
                zip(self.gradients, variables))
            tf.summary.scalar(
                'global_norm', global_norm)

        self.merged = tf.summary.merge_all()
        self.loss = loss

    def train(self, tensorboard_path, save_path, series_idx=None):
        series_idx_str = 'series_'+str(series_idx)+'_' if series_idx is not None else ''
        writer = tf.summary.FileWriter(
            '%s/%s' % (tensorboard_path, series_idx_str+datetime.now().strftime("%d:%m:%y-%H:%M:%S")), sess.graph)

        min_glob_loss = (1e99, -1)
        run = 0
        converged = False

        if self.batch_dims * self.p_val >= self.target_dims:
            replace_bool = True
        else:
            replace_bool = False

        pre_train_count = 0
        num_epochs = 10000
        for epoch in (range(num_epochs)):
            # print('Epoch:', epoch)
            batch_select = np.random.choice(
                np.arange(0, self.target_dims, self.batch_dims), size=self.p_val, replace=replace_bool)

            # batch_select = np.random.choice(
            #     np.arange(0, self.target_dims-self.batch_dims+1), size=self.p_val, replace=replace_bool)

            obs_pad_feats = []
            for item in self.obs_pad_store:
                obs_pad_feats.append(np.concatenate([np.reshape(
                    item[index: index + self.kernel_ext], [1, -1, 1]) for index in self.flow_dims * batch_select], axis=0))
            feat1 = np.concatenate(obs_pad_feats, axis=2)
            feat2 = np.concatenate([np.reshape(
                self.bin_feats[index: index + self.kernel_ext], [1, -1, 1]) for index in self.flow_dims * batch_select], axis=0)
            feat3 = np.concatenate([np.reshape(
                self.time_pad[index: index + self.kernel_ext], [1, -1, 1]) for index in self.flow_dims * batch_select], axis=0)
            feat4 = np.concatenate([np.reshape(
                self.time_till[index: index + self.kernel_ext], [1, -1, 1]) for index in self.flow_dims * batch_select], axis=0)

            time_feats_feed = np.concatenate(
                [feat1, feat2, feat3, feat4], axis=2)

            mask_feed = np.concatenate([np.expand_dims(self.mask_vals[:, index:(
                index + self.batch_dims + 1)], 0) for index in batch_select], axis=0)
            shift_feed = np.concatenate([np.expand_dims(self.shift_vals[:, index:(
                index + self.batch_dims + 1)], 0) for index in batch_select], axis=0)
            bin_feed = np.concatenate([np.expand_dims(
                self.obs_bin[:, index:index + self.batch_dims], 0) for index in batch_select], 0)

            if self.pre_train:
                if run == 0:
                    print("Initialising paths and parameters...")
                _, test, lf_sample = sess.run([self.t1, self.lf_log_prob, self.lf_sample], feed_dict={
                    self.time_feats: time_feats_feed, self.mask: mask_feed, self.shift: shift_feed, self.bin_feed: bin_feed, self.diffusivity: [0.0]})
                if np.sum(np.isinf(test)) == 0:
                    pre_train_count += 1
                else:
                    pre_train_count = 0
                if pre_train_count == 1000:
                    self.pre_train = False
                    print("Finished pre-training...")
                    run = 0

            else:
                _, summary, batch_loss = sess.run([self.train_step, self.merged, self.mean_loss], feed_dict={self.time_feats: time_feats_feed,
                                                                                                             self.mask: mask_feed, self.shift: shift_feed, self.bin_feed: bin_feed, self.diffusivity: [0.0]})
                
                # (sde_neg_sample, y_loc, y_scale, lf_sample_flat, sde_mu, sde_chol, xt_plus_dt, sde_drift,sde_sample,
                #  obs_eval, _, summary, batch_loss) = sess.run([
                #      self.sde_neg_sample, self.y_loc, self.y_scale, self.lf_sample_flat, self.sde_mu, self.sde_chol, self.xt_plus_dt, self.sde_drift,self.sde_sample,
                #     self.obs_eval, self.train_step, self.merged, self.mean_loss], feed_dict={self.time_feats: time_feats_feed,
                #                                                                                              self.mask: mask_feed, self.shift: shift_feed, self.bin_feed: bin_feed, self.diffusivity: [0.0]})
                # print('negative obs_eval', (obs_eval<1).sum())
                # print('negative y_loc:', (y_loc<1).sum())
                # print('negative y_scale:', (y_scale<0).sum())
                # print('lf_sample_flat:', lf_sample_flat.shape)
                # print(lf_sample_flat)
                # print('xt_plus_dt:', (xt_plus_dt<=1).sum()) # (150, 2)
                # # print('sde_drift:', sde_drift) # (150, 2)
                # print('negative lf_sample_flat:', (lf_sample_flat<=1).sum()) # (150, 2)
                # print('negative sde_sample:', (sde_sample<=1).sum()) # (150, 2)
                # print(sde_sample)
                # print('sde_sample from normal:') # (150, 2)
                # print(sde_neg_sample)
                # print('lf_sample_flat:')
                # print(lf_sample_flat)
                
                # print('sde_mu:', sde_mu.shape)
                # print(sde_mu)
                # neg_mu = sde_mu <= 1 #(150, 2)
                # print('negative sde_mu:', (neg_mu).sum())
                # # if neg_mu.sum() > 0:
                #     print('sde_mu:', sde_mu[neg_mu])
                #     print('xt_plus_dt:', xt_plus_dt[neg_mu])
                #     print('sde_drift:', sde_drift[neg_mu])
                
                # print('sde_chol:', sde_chol.shape)
                # print(sde_chol)
                # print('negative sde_chol:', (sde_chol[:,0,0]<0).sum() + (sde_chol[:,1,1]<0).sum()) # (150, 2)
                writer.add_summary(summary, run)

            if run % 1000 == 0:
                self.save(save_path)

            run += 1
            # print('='*50)
            # print()
            
            if epoch == num_epochs - 1:
                lf_sample_val, theta_val = sess.run([self.lf_sample, self.theta], feed_dict={
                    self.time_feats: time_feats_feed, self.mask: mask_feed, self.shift: shift_feed, self.bin_feed: bin_feed, self.diffusivity: [0.0]})
                np.save(f'dat/our_files/{series_idx_str}learned_lf_sample.npy', lf_sample_val)
                np.save(f'dat/our_files/{series_idx_str}learned_theta.npy', theta_val)
                for i in range(100):
                    lf_sample_init, lf_log_prob_init = self.SSM.slp()
                    lf_sample_neg = tf.transpose(tf.reshape(
                        lf_sample_init, [self.p_val, -1, 2]), [0, 2, 1])
                    qx_transform = tfb.Chain([
                        tfb.Affine(shift=[1.,1.], scale_diag=[1.,1.]),  
                        tfb.Softplus(event_ndims=2)                   
                        # tfb.Affine(shift=[-1.,-1.], scale_diag=[1.,1.])   
                        ])
                    def trp(x):
                        return tf.transpose(x, [0, 2, 1])
                    lf_sample_neg = trp(lf_sample_neg)
                    lf_sample = trp(qx_transform.forward(lf_sample_neg)) * self.mask + self.shift
                    lf_sample_val = sess.run(lf_sample, feed_dict={
                        self.time_feats: time_feats_feed, self.mask: mask_feed, self.shift: shift_feed, self.bin_feed: bin_feed, self.diffusivity: [0.0]})
                    np.save(f'dat/our_files/{series_idx_str}learned_lf_sample_%d.npy' % i, lf_sample_val)
                    

    def save(self, PATH):
        saver = tf.train.Saver()
        saver.save(sess, PATH)
        print("Model saved")

    def load(self, PATH):
        self.pre_train = False
        saver = tf.train.Saver()
        saver.restore(sess, PATH)
        print("Model restored")

    def save_paths(self, PATH_obs):

        path_store = []
        # 10 batches of length M=batch_dims=50 in  N=T/dt = 500
        for index_temp in (np.arange(0, self.target_dims, self.batch_dims)):

            batch_select = np.tile(index_temp, self.p_val)

            obs_pad_feats = []
            for item in self.obs_pad_store:
                obs_pad_feats.append(np.concatenate([np.reshape(
                    item[index: index + self.kernel_ext], [1, -1, 1]) for index in self.flow_dims * batch_select], axis=0))
            feat1 = np.concatenate(obs_pad_feats, axis=2) # (p_val, kernel_ext, feat_window)
            feat2 = np.concatenate([np.reshape(
                self.bin_feats[index: index + self.kernel_ext], [1, -1, 1]) for index in self.flow_dims * batch_select], axis=0)
            feat3 = np.concatenate([np.reshape(
                self.time_pad[index: index + self.kernel_ext], [1, -1, 1]) for index in self.flow_dims * batch_select], axis=0)
            feat4 = np.concatenate([np.reshape(
                self.time_till[index: index + self.kernel_ext], [1, -1, 1]) for index in self.flow_dims * batch_select], axis=0)
            # print('feat1:', feat1.shape)
            # print('feat2:', feat2.shape)
            # print('feat3:', feat3.shape)
            # print('feat4:', feat4.shape)
            time_feats_feed = np.concatenate(
                [feat1, feat2, feat3, feat4], axis=2)

            mask_feed = np.concatenate([np.expand_dims(self.mask_vals[:, index:(
                index + self.batch_dims + 1)], 0) for index in batch_select], axis=0)
            shift_feed = np.concatenate([np.expand_dims(self.shift_vals[:, index:(
                index + self.batch_dims + 1)], 0) for index in batch_select], axis=0)
            bin_feed = np.concatenate([np.expand_dims(
                self.obs_bin[:, index:index + self.batch_dims], 0) for index in batch_select], 0)

            path_out = sess.run(self.lf_sample, feed_dict={
                self.time_feats: time_feats_feed, self.mask: mask_feed, self.shift: shift_feed, self.bin_feed: bin_feed, self.diffusivity: [0.0]})
            ## lf_sample: [50, 2, 51]= [p, 2, M+1]
            path_store.append(path_out[:, :, 1:])

        paths = np.concatenate(path_store, axis=2) # (50, 2, 500)

        f = open(PATH_obs, 'w')
        # Reshape paths to (50, 1000)
        #Saving x values
        np.savetxt(f, np.reshape(paths, (self.p_val, -1)))
        f.close()


########### setting up the model ###########
# hyperparams
p_val = 1 #151  #number of random batches to be picked from time series
kernel_len = 20
dt = 0.2
T = 30 #8192
target_dims = np.int32(T / dt) + 1
batch_dims = 151 #length M of time series for partitioning the data
network_dims = [50] * 5
no_flows = 3 # number of flow layers
# priors = [(0., 3.0), (0.0, 3.0), (0.0, 3.0)]
#theta prior mean and scale
### Init theta and theta* priors: divided by 10?
priors = [(-1.0, np.sqrt(0.1)), (-6.0, np.sqrt(0.1)), 
          (-1.0, np.sqrt(0.1)), (-2.0, np.sqrt(0.1))]
feat_window = 10 #l' obs window size
print('\n'*3)
# obs and theta
x0_mean = np.array([91., 99.], dtype=NP_DTYPE)
x0_std = np.array([1., 1.], dtype=NP_DTYPE)


# f1 = open('dat/LV_obs_partial.txt', 'r')
# f2 = open('dat/LV_obs_binary.txt', 'r')
# f3 = open('dat/LV_time_till.txt', 'r')
f1 = open('dat/our_files/LV_obs_partial_test.txt', 'r')
f2 = open('dat/our_files/LV_obs_binary_test.txt', 'r')
f3 = open('dat/our_files/LV_time_till_test.txt', 'r')
# f1 = open('dat/small/LV_obs_partial_0.txt', 'r')
# f2 = open('dat/small/LV_obs_binary_0.txt', 'r')
# f3 = open('dat/small/LV_time_till_0.txt', 'r')

obs = np.loadtxt(f1, NP_DTYPE)
obs_not_observed = np.log(1 + np.exp(-2)) + 1.0 #f(x) = 1 + softplus(x-1) for obs=-1
print('obs_not_observed:', obs_not_observed)
obs[obs==-1] = obs_not_observed
obs_bin = np.loadtxt(f2, NP_DTYPE)
time_till = np.loadtxt(f3, NP_DTYPE)
f1.close()
f2.close()
f3.close()

print('obs:', obs.shape)
print('obs_bin:', obs_bin.shape)
print('time_till:', time_till.shape)


for i in tqdm(range(1, 1024)):
    obs = obs[:,i*target_dims:(i+1)*target_dims]
    obs_bin = obs_bin[:,i*target_dims:(i+1)*target_dims]
    time_till = time_till[:,i*target_dims:(i+1)*target_dims]

    # theta dist q(theta)
    bijectors = []
    num_bijectors = 4
    for j in range(num_bijectors):
        bijectors.append(tfb.Invert(tfb.MaskedAutoregressiveFlow(
            shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
                hidden_layers=[5, 5, 5], activation=tf.nn.elu))))
        if j < (num_bijectors - 1):
            bijectors.append(tfb.Permute(
                permutation=np.random.permutation(np.arange(0, len(priors)))))
    bijectors.append(tfb.Softplus(event_ndims=2))
    flow_bijector = tfb.Chain(list(reversed(bijectors)))
    print('\nflow_bijector:', flow_bijector.bijectors)
    theta_dist = tfd.TransformedDistribution(
        distribution=tfd.Normal(loc=0., scale=1.),
        bijector=flow_bijector,
        event_shape=[len(priors)])
    print('\ntheta_dist:', theta_dist)
    # buiding the model
    var_model = VI_SSM(obs, obs_bin, time_till, x0_mean, x0_std, theta_dist, priors, dt, T, p_val,
                    kernel_len, batch_dims, network_dims, target_dims, no_flows, feat_window, learn_rate = 1e-3, pre_train=True)
    var_model.build_flow()
    print('Model built')
    # new session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    var_model.save_paths(f'locally_variant/LV_obs_paths_series_{i}.txt')
    print('Paths saved')
    print('Training...')
    var_model.train(tensorboard_path='locally_variant/train/',
                    save_path=f'model_saves/LV_model_series_{i}_{batch_dims}_3.ckpt', series_idx=i)
