from random import seed

import numpy as np

class DLM_utility:
    def vectorize_seq(self, seq):
        try:
            seq[0][0]
        except TypeError:
            return self._make_vectorize_seq(seq)
        else:
            return seq

    def _make_vectorize_seq(self, one_dim_seq: list):
        return [[x] for x in one_dim_seq]

class DLM_D0_container:
    # notation
    # obser Y_t = F'_t \theta_t + v_t
    # state \theta_t = G_t \theta_{t-1} + w_t
    # error v_t ~ N(0, V_t), w_t~N(0, W_t). mutually indep, internally indep
    # mean.resp \mu = F'_t \theta_t
    def __init__(self, y_length: int):
        self.y_len = y_length

        self.F_obs_eq_design = None
        self.G_sys_eq_transition = None
        self.V_obs_eq_covariance = None
        self.W_sys_eq_covariance = None

        #optional
        self.u_covariate = None
        self.u_coeff_obs_eq_seq = None
        self.u_coeff_state_eq_seq = None


    # == setters: with an input as a sequence ==
    def set_Vt_obs_eq_covariance(self, Vt_seq: list[np.array]):
        self.V_obs_eq_covariance = Vt_seq

    def set_Wt_state_error_cov(self, Wt_seq: list[np.array]):
        self.W_sys_eq_covariance = Wt_seq

    def set_Ft_design_mat(self, Ft_seq: list[np.array]):
        self.F_obs_eq_design = Ft_seq

    def set_Gt_transition_mat(self, Gt_seq: np.array):
        self.G_sys_eq_transition = Gt_seq

    def set_ut_covariate_and_coeff(self,
            ut_covariates_seq: list[np.array],
            obs_eq_coeff_seq: list[np.array], state_eq_coeff_seq: list[np.array]):
        self.u_covariate = ut_covariates_seq
        self.u_coeff_obs_eq_seq = obs_eq_coeff_seq
        self.u_coeff_state_eq_seq = state_eq_coeff_seq

    # == setters: with an input as a point (when the value is constant on time in the model) ==
    def set_V_const_obs_eq_covariance(self, V: np.array):
        self.V_obs_eq_covariance = [V for _ in range(self.y_len)]

    def set_W_const_state_error_cov(self, W: np.array):
        self.W_sys_eq_covariance = [W for _ in range(self.y_len)]

    def set_F_const_design_mat(self, F: np.array):
        self.F_obs_eq_design = [F for _ in range(self.y_len)]

    def set_G_const_transition_mat(self, G: np.array):
        self.G_sys_eq_transition = [G for _ in range(self.y_len)]

    def set_u_const_covariate_and_coeff(self,
            u: np.array,
            obs_eq_coeff: np.array, state_eq_coeff: np.array):
        self.u_covariate = [u for _ in range(self.y_len)]
        self.u_coeff_obs_eq_seq = [obs_eq_coeff for _ in range(self.y_len)]
        self.u_coeff_state_eq_seq = [state_eq_coeff for _ in range(self.y_len)]
    # == end setters ==

    def set_u_no_covariate(self):
        self.set_u_const_covariate_and_coeff(np.array([0]), np.array([0]), np.array([0]))



class DLM_simulator:
    def __init__(self, D0: DLM_D0_container, set_seed=None):
        self.D0 = D0

        self.theta_seq = []
        self.y_seq = []
        if set_seed is not None:
            seed(set_seed)
            self.rv_generator = np.random.default_rng(seed=set_seed)
        else:
            self.rv_generator = np.random.default_rng()

    def simulate_data(self, initial_m0, initial_C0):
    #                       E(theta_0|D0), var(theta_0|D0)
        theta_0 = self.rv_generator.multivariate_normal(initial_m0, initial_C0)
        self.theta_seq.append(theta_0)
        for t in range(self.D0.y_len):
            theta_last = self.theta_seq[-1]
            theta_t = self.D0.G_sys_eq_transition[t] @ theta_last + \
                    self.rv_generator.multivariate_normal(np.zeros(theta_last.shape), self.D0.W_sys_eq_covariance[t])
            y_t = np.transpose(self.D0.F_obs_eq_design[t]) @ theta_t + \
                    self.rv_generator.multivariate_normal(np.zeros(self.D0.V_obs_eq_covariance[t].shape[0]), self.D0.V_obs_eq_covariance[t])

            self.theta_seq.append(theta_t)
            self.y_seq.append(y_t)
    
    def get_theta_y(self):
        return self.theta_seq[1:], self.y_seq
