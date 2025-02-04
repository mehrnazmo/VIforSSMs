import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
from tensorflow.python.client import timeline
from tensorflow.python.ops import clip_ops
from optimisers.adamax import AdamaxOptimizer
from tqdm import tqdm
import traceback

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
        sample = self.sample(p_val)     #62+T
        log_prob = tf.reduce_sum(self.log_prob(
            sample)[:, -self.batch_dims:], axis=1)
        return sample, log_prob


class Bivariate_Normal():
    '''
    bivariate batched normal dist
    '''

    def __init__(self, mu, chol):
        self.mu = tf.expand_dims(mu, 2)
        self.chol = chol + tf.expand_dims(tf.eye(2) * 1e-6, 0) # (p*(T-1), 2, 2)
        self.det = tf.reduce_prod(tf.matrix_diag_part(chol), axis=1) ** 2
        self.cov_inv = tf.matrix_inverse(chol @ tf.transpose(chol, [0, 2, 1]))

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
        # print('base_sample:', base_sample.shape)
        # print('base_logprob:', self.base_logprob.shape)
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
        base_dims = kernel_len * no_flows + batch_dims + 2 #62+T
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

    def __init__(self, obs, obs_bin, time_till, x0_mean, x0_std, theta_dist, fix_theta, sparse, priors, dt, T, p_val, kernel_len, batch_dims, network_dims, target_dims, no_flows, feat_window, learn_rate = 1e-3, pre_train=True):
        # raw inputs -> class variables
        self.obs = obs #(1024,2,151)
        self.obs_bin = obs_bin #(1024,2,151)
        self.bs = obs.shape[0]
        obs_flatten = np.reshape(obs, (obs.shape[0], -1), order='F') #(bs,2*151)
        self.flow_dims = 2
        self.fix_theta = fix_theta
        self.sparse = sparse
        self.sparse_str = 'sparse' if sparse else 'dense'
        self.theta_dist = theta_dist
        if self.fix_theta:
            self.theta = tf.convert_to_tensor(np.tile(np.array(priors), [p_val, 1]), DTYPE) # (p, 4)
            self.theta_log_prob = tf.constant(0.0)
        else:    
            self.theta = theta_dist.sample(p_val)
            self.theta_log_prob = theta_dist.log_prob(self.theta) # log q(theta)
        self.priors = priors
        self.dt = dt
        self.p_val = p_val
        self.kernel_len = kernel_len
        self.batch_dims = batch_dims
        self.network_dims = network_dims
        self.target_dims = target_dims
        self.no_flows = no_flows
        self.theta_eval = self._theta_strech() #4: (p*(T-1),)
        self.diffusivity = tf.placeholder(DTYPE, 1)
        self.pre_train = pre_train
        self.learn_rate = learn_rate
        self.kernel_ext = self.kernel_len * self.no_flows + self.flow_dims * self.batch_dims + 2 #62+2*T
        # augementing raw inputs
        self.obs_pad_store = [] #l', bs, 62+2pT
        for i in range(0, feat_window*5, 5):
            self.obs_pad_store.append(np.concatenate(
                (np.zeros((self.bs, self.no_flows * self.kernel_len + self.flow_dims - i)), obs_flatten, np.zeros((self.bs,i))), axis=1))
        self.time_pad = np.concatenate((np.zeros(self.no_flows * self.kernel_len + self.flow_dims),
                                        np.repeat(np.arange(0, T + dt, dt), self.flow_dims)), axis=0) #62+2pT #TODO
        time_till_pad = np.reshape(np.repeat(np.arange(np.round((self.no_flows * self.kernel_len + self.flow_dims) * (
            self.dt / self.flow_dims), 1), 0., -self.dt), self.flow_dims), (self.flow_dims, -1), 'F') #(2, 31)
        self.time_till = np.reshape(np.concatenate(
            (np.broadcaset_to(time_till_pad, (self.bs, *time_till_pad.shape)), time_till), 2), (self.bs, -1), 'F') #bs, 62+2pT
        
        self.bin_feats = np.float32(np.concatenate(
            (np.zeros(self.no_flows * self.kernel_len + self.flow_dims), 
             np.ones(self.target_dims * self.flow_dims)), axis=0))  #62+2pT #TODO
        
        self.mask_vals = np.concatenate(
            (np.zeros((2, 1)), np.ones((self.flow_dims, self.target_dims))), axis=1) #(2,p(T+1)) #TODO
        self.shift_vals = np.concatenate(
            (np.expand_dims(x0_mean, 1), np.zeros((self.flow_dims, self.target_dims))), axis=1)#(2,p(T+1)) #TODO
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

        # model placeholders #TODO
        self.time_feats = tf.placeholder(  #(p, 62+2*T, 13)
            shape=[self.p_val, self.kernel_len * self.no_flows + self.flow_dims * self.batch_dims + self.flow_dims, feat_window + 3], dtype=DTYPE)
        self.obs_eval = tf.transpose(tf.reshape( #(p, 2, T)
            self.time_feats[:, -self.flow_dims * self.batch_dims:, 0], [self.p_val, -1, 2]), [0, 2, 1]) #get only obs
        self.mask = tf.placeholder( #(p, 2, T+1)
            shape=[self.p_val, self.flow_dims, self.batch_dims + 1], dtype=DTYPE)
        self.shift = tf.placeholder( #(p, 2, T+1)
            shape=[self.p_val, self.flow_dims, self.batch_dims + 1], dtype=DTYPE)
        self.bin_feed = tf.placeholder( #(p, 2, T)
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
        #used batch_dims-1 instead of batch_dims to exclude x0
        slice_stash = []
        for i in range(len(self.priors)):
            slice_stash.append(tf.reshape(tf.tile(tf.expand_dims(
                (self.theta[:, i]), 1), [1, self.batch_dims-1]), [-1])) 
        return slice_stash

    def _ELBO(self):
        #y_loc: (p, 2, T), theta: (p, 4), lf_sample: # (p, 2, T+1)
        self.y_loc = self.lf_sample[:, :, 1:]
        self.y_scale = (tf.expand_dims(tf.expand_dims(self.theta[:,-1], 1), 2) 
                        * self.lf_sample[:, :, 1:]) # (p, 2, T+1)
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
        
        #p(x) except for x0
        flow_head = self.lf_sample[:, :, 1:-1] # (p, 2, T-1)
        lf_sample_flat = tf.concat(
            [tf.reshape(self.lf_sample[:, 0, 2:], [-1, 1]),
            tf.reshape(self.lf_sample[:, 1, 2:], [-1, 1])], 1) #(p*(T-1), 2)
        x_t = tf.concat([tf.reshape(flow_head[:, 0, :], [-1, 1]),
                         tf.reshape(flow_head[:, 1, :], [-1, 1])], 1) #(p*(T-1), 2)
        self.sde_drift = self.dt * alpha(tf.reshape(flow_head[:, 0, :], [-1]), 
                                         tf.reshape(flow_head[:, 1, :], [-1]), 
                                         self.theta_eval) ##theta_eval: 4 * (p*(T-1),)  # (p*(T-1), 2)
        self.sde_mu = self.sde_drift + x_t ## (p*(T-1), 2)
        self.sde_chol = tf.sqrt(self.dt) * sqrt_beta(
            tf.reshape(flow_head[:, 0, :], [-1]), 
            tf.reshape(flow_head[:, 1, :], [-1]), 
            self.theta_eval) ##theta_eval: 4 * (p*(T-1),)   # (p*(T-1), 2, 2)

        SDE_log_prob = tf.reduce_sum(tf.reshape(
            Bivariate_Normal(mu=self.sde_mu, chol=self.sde_chol).log_prob(
                lf_sample_flat, 
                bijector_chain=tfb.Chain([
                    tfb.Affine(shift=[1.,1.], scale_diag=[1.,1.]),  
                    tfb.Softplus(event_ndims=2),                      
                    tfb.Affine(shift=[-1.,-1.], scale_diag=[1.,1.])])
                ), [self.p_val, -1]), 1) # (p,)

        
        ## p(x0)
        x0_sample = self.lf_sample[:, :, 1] # (p, 2)
        x0_transforms = tfb.Chain([
            tfb.Affine(shift=[1.,1.], scale_diag=[1.,1.]),  
            tfb.Softplus(event_ndims=2),                      
            tfb.Affine(shift=[-1.,-1.], scale_diag=[1.,1.])    
                ])
        x0_dist_init = tfd.MultivariateNormalDiag(loc=self.x0_mean, scale_diag=self.x0_std)
        x0_dist = tfd.TransformedDistribution(distribution=x0_dist_init, bijector=x0_transforms)
        self.x0_log_prob = x0_dist.log_prob(x0_sample)
        SDE_log_prob += self.x0_log_prob
        
        #log p(x) - log q(x) + log p(y|x) + log p(theta) - log q(theta)
        if self.fix_theta:
            prior_log_prob = tf.constant(0.0)
        else:
            prior_mean = [item[0] for item in self.priors]
            prior_scale = [item[1] for item in self.priors]
            # log p(theta)
            prior_log_prob = tfd.MultivariateNormalDiag(
                loc=prior_mean, scale_diag=prior_scale).log_prob(self.theta)

        ELBO = (self.target_dims / self.batch_dims) * (
            SDE_log_prob - self.lf_log_prob + obs_log_prob) + prior_log_prob - self.theta_log_prob #all have dimension (p,)

        # ELBO = (self.target_dims / self.batch_dims) * (SDE_log_prob - self.lf_log_prob) + prior_log_prob - self.theta_log_prob
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
        #lf_sample_init: x samples output of q(x|theta) with IAF (p_val, 102)
        # lf_sample_neg: (p_val, (u,v)=2, M+1=51)
        #lf_log_prob_init: log q(x) (p,)
        lf_sample_init, lf_log_prob_init = self.SSM.slp() #slp: sample, log probability
        self.lf_sample_neg = tf.transpose(tf.reshape(
            lf_sample_init, [self.p_val, -1, 2]), [0, 2, 1]) #lf_sample_neg: (p,2,T+1)
        # transform
        qx_transform = tfb.Chain([
            tfb.Affine(shift=[1.,1.], scale_diag=[1.,1.]),  
            tfb.Softplus(event_ndims=2)                   
        ])
        
        def trp(x):
            return tf.transpose(x, [0, 2, 1])
        #self.lf_sample (p, 2, T+1): p batches of
        # [[u0, softplus(u1_q), ..., softplus(uM_q)], 
        #  [v0, softplus(v1_q), ..., softplus(vM_q)]]
        lf_sample_neg = trp(self.lf_sample_neg) # (p, T+1, 2)
        #modifying log q(x) to accomodate softplus
        self.lf_sample = trp(qx_transform.forward(lf_sample_neg)) * self.mask + self.shift # (p, 2, T+1)
        
        self.lf_log_prob = lf_log_prob_init + qx_transform.inverse_log_det_jacobian(
            trp(self.lf_sample[:, :, 1:])) #log q(x) from normalizing flow # (p,)       
        loss, self.sde_loss, self.obs_loss, prior_prob = self._ELBO()
        self.mean_loss = tf.reduce_mean(loss)
        
        self.t1 = AdamaxOptimizer(
            learning_rate=1e-3, beta1=0.9).minimize((self.lf_sample - 75) ** 2)

        # self.t2 = AdamaxOptimizer(learning_rate=1e-3, beta1=0.9).minimize(
            # (self.theta - tf.log(tf.tile(np.expand_dims(self.priors, axis=0), [self.p_val, 1]))) ** 2)
        # do something nicer with this!
        theta_pos_index = [True, True, True]
        with tf.name_scope('loss'):
            tf.summary.scalar('ELBO', self.mean_loss)
            tf.summary.scalar('NELBO', -self.mean_loss)
            tf.summary.scalar('Time', time.time())
            tf.summary.scalar(
                'SDE_log_prob p(x)', (self.target_dims / self.batch_dims) * tf.reduce_mean(self.sde_loss))
            if not self.fix_theta:
                tf.summary.scalar('theta_log_prob p(theta)', tf.reduce_mean(self.theta_log_prob))
                # tf.summary.scalar('truth_log_prob q(theta*)', -self.theta_dist.log_prob(np.expand_dims(self.priors, axis=0))[0])
                tf.summary.scalar('truth_log_prob q(theta*)', -self.theta_dist.log_prob([[0.31326169, 0.00247569, 0.31326169, 0.2]])[0])
            tf.summary.scalar(
                'obs_log_prob p(y|x)', (self.target_dims / self.batch_dims) * tf.reduce_mean(self.obs_loss))
            tf.summary.scalar(
                'path_log_prob q(x)', (self.target_dims / self.batch_dims) * tf.reduce_mean(self.lf_log_prob))
            # theta summaries
            for i in range(len(theta_pos_index)):
                if theta_pos_index[i]:
                    tf.summary.histogram(str(i),
                        self.theta[:, i], family='parameters')
                else:
                    tf.summary.histogram(
                        str(i), self.theta[:, i], family='parameters')
            
        with tf.name_scope('theta_update'):
            tf.summary.scalar('min chol 00', tf.reduce_min(self.sde_chol[:,0,0]))
            tf.summary.scalar('min chol 11', tf.reduce_min(self.sde_chol[:,1,1]))
            tf.summary.scalar('min chol 10', tf.reduce_min(self.sde_chol[:,1,0]))
            
            tf.summary.scalar('max chol 00', tf.reduce_max(self.sde_chol[:,0,0]))
            tf.summary.scalar('max chol 11', tf.reduce_max(self.sde_chol[:,1,1]))
            tf.summary.scalar('max chol 10', tf.reduce_max(self.sde_chol[:,1,0]))            
            

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

    def train(self, tensorboard_path, save_path, num_epochs, pre_train_epochs, series_idx=None):
        series_idx_str = 'series_'+str(series_idx)+'_' if series_idx is not None else ''
        writer = tf.summary.FileWriter(
            '%s/%s' % (tensorboard_path, series_idx_str+datetime.now().strftime("%d:%m:%y-%H:%M:%S")), sess.graph)

        min_glob_loss = (1e99, -1)
        run = 0
        converged = False

        # if self.batch_dims * self.p >= self.target_dims:
        #     replace_bool = True
        # else:
        #     replace_bool = False
        replace_bool = False

        pre_train_count = 0
        start_time = time.time() 
        for epoch in tqdm(range(num_epochs)):

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
                if pre_train_count == pre_train_epochs:
                    self.pre_train = False
                    print("Finished pre-training...")
                    run = 0

            else:
                _, summary, batch_loss = sess.run([self.train_step, self.merged, self.mean_loss], feed_dict={self.time_feats: time_feats_feed,
                                                                                                             self.mask: mask_feed, self.shift: shift_feed, self.bin_feed: bin_feed, self.diffusivity: [0.0]})
                writer.add_summary(summary, run)

            elapsed_time = time.time() - start_time  # Elapsed time since training started
            print(f"Epoch {epoch + 1}/{num_epochs}, Elapsed Time: {elapsed_time:.2f} seconds")
            elapsed_time_summary = tf.Summary(value=[tf.Summary.Value(tag="Elapsed Time", simple_value=elapsed_time)])
            writer.add_summary(elapsed_time_summary, run)
            
            if run % 1000 == 0:
                self.save(save_path)
            run += 1
            directory = os.path.dirname(save_path)
            if epoch == num_epochs - 1:
                self.save(save_path)
                lf_sample_val = sess.run([self.lf_sample], feed_dict={
                    self.time_feats: time_feats_feed, self.mask: mask_feed, self.shift: shift_feed, self.bin_feed: bin_feed, self.diffusivity: [0.0]})
                theta_str = 'fix_theta' if self.fix_theta else 'learn_theta'
                np.save(f'{directory}/lf_sample_val_{series_idx_str}.npy', lf_sample_val)
                for i in range(100):
                    lf_sample_init, lf_log_prob_init = self.SSM.slp()
                    lf_sample_neg = tf.transpose(tf.reshape(
                        lf_sample_init, [self.p_val, -1, 2]), [0, 2, 1])
                    qx_transform = tfb.Chain([
                        tfb.Affine(shift=[1.,1.], scale_diag=[1.,1.]),  
                        tfb.Softplus(event_ndims=2)                   
                        ])
                    def trp(x):
                        return tf.transpose(x, [0, 2, 1])
                    lf_sample_neg = trp(lf_sample_neg)
                    lf_sample = trp(qx_transform.forward(lf_sample_neg)) * self.mask + self.shift
                    lf_sample_val = sess.run(lf_sample, feed_dict={
                        self.time_feats: time_feats_feed, self.mask: mask_feed, self.shift: shift_feed, self.bin_feed: bin_feed, self.diffusivity: [0.0]})
                    np.save(f'{directory}/lf_sample_val_series_{series_idx_str}_{i}.npy', lf_sample_val)
                    

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
        # only one batch with index_temp = 0 if batch_dims = N
        for ii, index_temp in tqdm(enumerate(np.arange(0, self.target_dims, self.batch_dims))):
            batch_select = np.tile(index_temp, self.p_val)

            obs_pad_feats = []
            for item in self.obs_pad_store: ##obs_pad_store: l', 62+2pT  ##kernel_ext: 62+2*T
                obs_pad_feats.append(np.concatenate([np.reshape(
                    item[index: index + self.kernel_ext], [1, -1, 1]) for index in self.flow_dims * batch_select], axis=0))
            feat1 = np.concatenate(obs_pad_feats, axis=2) # (p_val, kernel_ext=62+2T, feat_window)
            feat2 = np.concatenate([np.reshape( # (p_val, kernel_ext=62+2T, 1)
                self.bin_feats[index: index + self.kernel_ext], [1, -1, 1]) for index in self.flow_dims * batch_select], axis=0)
            feat3 = np.concatenate([np.reshape(  # (p_val, kernel_ext=62+2T, 1) ##time_pad: 62+2T
                self.time_pad[index: index + self.kernel_ext], [1, -1, 1]) for index in self.flow_dims * batch_select], axis=0)
            feat4 = np.concatenate([np.reshape(  # (p_val, kernel_ext=62+2T, 1)
                self.time_till[index: index + self.kernel_ext], [1, -1, 1]) for index in self.flow_dims * batch_select], axis=0)
            print('saving paths...')
            
            time_feats_feed = np.concatenate(
                [feat1, feat2, feat3, feat4], axis=2) # (p_val, kernel_ext=62+2T, l'+3=13)

            mask_feed = np.concatenate([np.expand_dims(self.mask_vals[:, index:(
                index + self.batch_dims + 1)], 0) for index in batch_select], axis=0) # (p_val, 2, T+1)
            shift_feed = np.concatenate([np.expand_dims(self.shift_vals[:, index:(
                index + self.batch_dims + 1)], 0) for index in batch_select], axis=0) # (p_val, 2, T+1)
            bin_feed = np.concatenate([np.expand_dims(
                self.obs_bin[:, index:index + self.batch_dims], 0) for index in batch_select], 0) # (p_val, 2, T)

            path_out = sess.run(self.lf_sample, feed_dict={
                self.time_feats: time_feats_feed, self.mask: mask_feed, self.shift: shift_feed, self.bin_feed: bin_feed, self.diffusivity: [0.0]})
            ## lf_sample: [50, 2, 51]= [p, 2, T]
            path_store.append(path_out[:, :, 1:])

        paths = np.concatenate(path_store, axis=2) # # (p_val, 2, T)
        f = open(PATH_obs, 'w')
        #Saving x values
        np.savetxt(f, np.reshape(paths, (self.p_val, -1))) # (p_val, 2*T)
        f.close()


########### setting up the model ###########
# hyperparams
#theta prior 
### Init theta and theta* priors: divided by 10?
def softplus_np_(x):
    return np.log(1 + np.exp(x))
priors = softplus_np_([-1.0, -6.0, -1.0, np.log(np.exp(0.2)-1)])
priors[-1] = 0.2

# theta dist
bijectors = []
num_bijectors = 4
for i in range(num_bijectors):
    bijectors.append(tfb.Invert(tfb.MaskedAutoregressiveFlow(
        shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
            hidden_layers=[5, 5, 5], activation=tf.nn.elu))))
    if i < (num_bijectors - 1):
        bijectors.append(tfb.Permute(
            permutation=np.random.permutation(np.arange(0, len(priors)))))
flow_bijector = tfb.Chain(list(reversed(bijectors)))

theta_dist = tfd.TransformedDistribution(
    distribution=tfd.Normal(loc=0., scale=1.),
    bijector=flow_bijector,
    event_shape=[len(priors)])

# theta_dist = tfd.MultivariateNormalDiag(loc = [tf.Variable(0.05), tf.Variable(.05), tf.Variable(0.05)], scale_diag= [tf.Variable(1.), tf.Variable(1.), tf.Variable(1.)])

try:
    tf.reset_default_graph()
    p_val = 1 #3  #number of random batches to be picked from time series
    kernel_len = 20
    dt = 0.2
    T = 30
    target_dims = np.int32(T / dt) + 1
    batch_dims = 151 #length M of time series for partitioning the data
    network_dims = [50] * 5
    no_flows = 3 # number of flow layers
    num_epochs = 1600
    pre_train_epochs = 1000
    fix_theta = True
    feat_window = 10 #l' obs window size
    print('\n'*3)
    # obs and theta
    x0_mean = np.array([91., 99.], dtype=NP_DTYPE)
    x0_std = np.array([1., 1.], dtype=NP_DTYPE)
    
    # f1 = open('dat/LV_obs_partial.txt', 'r')
    # f2 = open('dat/LV_obs_binary.txt', 'r')
    # f3 = open('dat/LV_time_till.txt', 'r')
    # f1 = open('dat/our_files/LV_obs_partial_test.txt', 'r')
    # f2 = open('dat/our_files/LV_obs_binary_test.txt', 'r')
    # f3 = open('dat/our_files/LV_time_till_test.txt', 'r')
    # f1 = open('dat/small/LV_obs_partial_0.txt', 'r')
    # f2 = open('dat/small/LV_obs_binary_0.txt', 'r')
    # f3 = open('dat/small/LV_time_till_0.txt', 'r')
    
    f1 = open('./dat/our_files/fix_theta/LV_obs_partial_dense_test.txt', 'r')
    f2 = open('./dat/our_files/fix_theta/LV_obs_binary_dense_test.txt', 'r')
    f3 = open('./dat/our_files/fix_theta/LV_time_till_dense_test.txt', 'r')

    obs = np.loadtxt(f1, NP_DTYPE)
    obs_not_observed = np.log(1 + np.exp(-2)) + 1.0 #f(x) = 1 + softplus(x-1) for obs=-1
    print('obs_not_observed:', obs_not_observed)
    obs[obs==-1] = obs_not_observed
    obs_bin = np.loadtxt(f2, NP_DTYPE)
    time_till = np.loadtxt(f3, NP_DTYPE)
    f1.close()
    f2.close()
    f3.close()

    print('obs:', obs.shape) #(2,151*1024)
    print('obs_bin:', obs_bin.shape) #(2,151*1024)
    print('time_till:', time_till.shape) #(2,151*1024)
    
    obs = (obs.reshape(2, 1024, 151)).transpose(1, 0, 2) # (1024, 2, 151)
    obs_bin = obs_bin.reshape(2, 1024, 151).transpose(1, 0, 2) # (1024, 2, 151)
    time_till = time_till.reshape(2, 1024, 151).transpose(1, 0, 2) # (1024, 2, 151)

    sparse = True if 'sparse' in f1.name else False
    sparse_str = 'sparse' if sparse else 'dense'
    theta_str = 'fix_theta' if fix_theta else 'learn_theta'
    
    # buiding the model
    var_model = VI_SSM(obs, obs_bin, time_till, x0_mean, x0_std, theta_dist, fix_theta, sparse, priors, dt, T, p_val,
                    kernel_len, batch_dims, network_dims, target_dims, no_flows, feat_window, learn_rate = 1e-3, pre_train=True)
    var_model.build_flow()
    print('Model built')
    out_dir = f'~/gmvi/ryder_run/VIforSSMs/out/{time.strftime("%Y%m%d-%H%M%S")}_{sparse_str}_{theta_str}'
    os.makedirs(out_dir, exist_ok=True)
    # new session
    with tf.Session() as sess:  # or InteractiveSession if needed
        sess.run(tf.global_variables_initializer())
        var_model.save_paths(f'{out_dir}/lf_sample_series.txt')
        # print('Paths saved')
        # print('Training...')
        # np.savetxt('/home/b2028663/scripts/arf/locally_variant/local_post.txt', sess.run(theta_dist.sample([100000])))
        var_model.train(tensorboard_path=f'{out_dir}/tf_series',
                        save_path=f'{out_dir}/LV_model_series.ckpt',
                        num_epochs=num_epochs, pre_train_epochs=pre_train_epochs, series_idx=0)
except Exception as e:
    print('\nError:', e)
    print()
    print(traceback.print_exc())
    print()
    print('Error:', e)
    
print('All series done!')