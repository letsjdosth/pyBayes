import numpy as np

class KalmanFilter:
    # notation
    # obser Y_t = F'_t \theta_t + v_t
    # state \theta_t = G_t \theta_{t-1} + w_t
    # error v_t ~ N(0, V_t), w_t~N(0, W_t). mutually indep, internally indep
    # mean.resp \mu = F'_t \theta_t

    def __init__(self, observed_seq: list):
        self.y = self._vectorize_seq(observed_seq)
        self.T = len(self.y)

        #default
        self.theta_coeff_obs_eq_seq = None
        self.theta_coeff_state_eq_seq = None
        self.V_seq = None
        self.W_seq = None
        self.set_no_covariate()

        #result containers        
        self.E_theta_t_given_D_t_1 = [] #x_t^{t-1} in Shumway-Stoffer
        self.E_var_theta_t_given_D_t_1 = [] # P_t^{t-1} in Shumway-Stoffer
        self.innovation = []

    def _vectorize_seq(self, seq):
        try:
            seq[0][0]
        except TypeError:
            return self._make_vectorize_seq(seq)
        else:
            return seq
        
    def _make_vectorize_seq(self, one_dim_seq: list):
        return [[x] for x in one_dim_seq]

    # == setters ==
    #must run
    def set_obs_error_cov_with_time_varying(self, obs_error_cov_seq: list[np.array]):
        self.V_seq = obs_error_cov_seq
    #later: const version

    #must run
    def set_state_error_cov_with_time_varying(self, state_error_cov_seq: list[np.array]):
        self.W_seq = state_error_cov_seq
    #later: const version

    #must run either "set_coeff_with_time_varying" or "set_coeff_with_const"
    def set_coeff_with_time_varying(self, obs_eq_coeff_seq: list[np.array], state_eq_coeff_seq: list[np.array]):
        #                                 F_t                             , G_t
        self.theta_coeff_obs_eq_seq = obs_eq_coeff_seq
        self.theta_coeff_state_eq_seq = state_eq_coeff_seq

    def set_coeff_with_const(self, obs_eq_coeff_mat: np.array, state_eq_coeff_mat: np.array):
        #                          F_t=F                     , G_t=G
        self.theta_coeff_obs_eq_seq = [obs_eq_coeff_mat for _ in range(self.T)]
        self.theta_coeff_state_eq_seq = [state_eq_coeff_mat for _ in range(self.T)]

    #optionally run "set_covariate_with_time_varying_coeff" or "set_covariate_with_const_coeff"
    def set_covariate_with_time_varying_coeff(self, covariates_seq: list, obs_eq_coeff_seq: list[np.array], state_eq_coeff_seq: list[np.array]):
        self.u = self._vectorize_seq(covariates_seq)
        self.u_coeff_obs_eq_seq = obs_eq_coeff_seq
        self.u_coeff_state_eq_seq = state_eq_coeff_seq

    def set_covariate_with_const_coeff(self, covariates_seq: list, obs_eq_coeff_mat: np.array, state_eq_coeff_mat: np.array):
        self.u = self._vectorize_seq(covariates_seq)
        self.u_coeff_obs_eq_seq = [obs_eq_coeff_mat for _ in range(self.T)]
        self.u_coeff_state_eq_seq = [state_eq_coeff_mat for _ in range(self.T)]

    def set_no_covariate(self): #default
        self.u = [[0] for _ in range(self.T)]
        self.u_coeff_obs_eq_seq = [[0] for _ in range(self.T)]
        self.u_coeff_state_eq_seq = [[0] for _ in range(self.T)]
    # == end setters ==

    #filtering
    def run_filter(self, initial_state_mean_resp, initial_innovation_var):
        #                x_0^0 = \mu_0          , P_0^0 = \Sigma_0
        
        #check everything is set

        #initialize
        self.E_theta_0 = initial_state_mean_resp
        self.E_var_theta_0 = initial_innovation_var
        
        #run
        # for i in range(1, )
        # self._filter_oneiter()
    
    def _filter_oneiter(self):
        pass


if __name__=="__main__":
    y1 = [1,2,3,4,5]
    y2 = [[1],[2],[3],[4],[5]]
    y3 = [[1,0],[2,0],[3,0],[4,0],[5,0]]
    test_inst = KalmanFilter(y1)

