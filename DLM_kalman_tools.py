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


class KalmanFilter:
    def __init__(self, y_observation, D0: DLM_D0_container, initial_mu0_given_D0, initial_P0_given_D0):
        self.util_inst = DLM_utility()
        
        self.y_observation = self.util_inst.vectorize_seq(y_observation)
        self.D0 = D0
        self.mu0 = initial_mu0_given_D0
        self.P0 = initial_P0_given_D0

        #result containers
        self.theta_one_step_forecast = [] #x_t^{t-1} in Shumway-Stoffer, f_t=E(\theta_t|D_{t-1}) in West
        self.P_one_step_forecast = [] # P_t^{t-1} in Shumway-Stoffer, R_t=Var(\theta_t|D_{t-1}) in West

        self.theta_on_time = [initial_mu0_given_D0] #x_t^{t} in Shumway-Stoffer, m_t=E(\theta_t|D_{t}) in West
        self.P_on_time = [initial_P0_given_D0] # P_t^{t} in Shumway-Stoffer, C_t=Var(\theta_t|D_{t}) in West

        self.innovation = []
        self.innovation_cov = []
        self.kalman_gain = []

    #filtering
    def _filter_one_iter(self, t):
        # one-step forecast (prior)
        Gt = self.D0.G_sys_eq_transition[t-1]
        ut = self.D0.u_covariate[t-1]
        ut_sys_coeff = self.D0.u_coeff_state_eq_seq[t-1]
        theta_t_one_step_forecast = Gt @ self.theta_on_time[-1] + ut_sys_coeff @ ut
        Pt_one_step_forecast = Gt @ self.P_on_time[-1] @ np.transpose(Gt) + self.D0.W_sys_eq_covariance[t-1]

        # on_time (posterior)
        At = self.D0.F_obs_eq_design[t-1]
        Rt = self.D0.V_obs_eq_covariance[t-1]
        ut_obs_coeff = self.D0.u_coeff_obs_eq_seq[t-1]
        
        innovation_t = self.y_observation[t-1] - At @ theta_t_one_step_forecast - ut_obs_coeff @ ut
        innovation_t_cov = At @ Pt_one_step_forecast @ np.transpose(At) + Rt
        kalman_gain_t = Pt_one_step_forecast @ At @ np.linalg.inv(innovation_t_cov)

        theta_t_on_time = theta_t_one_step_forecast + kalman_gain_t @ innovation_t
        KtAt = kalman_gain_t @ At
        Pt_on_time = (np.identity(KtAt.shape[0]) - KtAt) @ Pt_one_step_forecast

        # save
        self.theta_one_step_forecast.append(theta_t_one_step_forecast)
        self.P_one_step_forecast.append(Pt_one_step_forecast)

        self.theta_on_time.append(theta_t_on_time)
        self.P_on_time.append(Pt_on_time)

        self.innovation.append(innovation_t)
        self.innovation_cov.append(innovation_t_cov)
        self.kalman_gain.append(kalman_gain_t)


    def run_filter(self):
        #check everything is set
        #update
        for t in range(self.D0.y_len):
            self._filter_one_iter(t+1)

        # delete innitial value
        self.theta_on_time = self.theta_on_time[1:]
        self.P_on_time = self.P_on_time[1:]

    def run_smoother(self, given_time=None):
        # if given_time is None:
        #     given_time = self.D0.y_len
        pass #impl later



if __name__=="__main__":
    import matplotlib.pyplot as plt
    test1 = False
    test2 = True

    if test1:
        test1_W = [np.array([[1]]) for _ in range(100)]
        test1_V = [np.array([[0.1]]) for _ in range(100)]
        test1_F = [np.array([[1]]) for _ in range(100)]
        test1_G = [np.array([[0.9]]) for _ in range(100)]
        test1_D0 = DLM_D0_container(100)
        test1_D0.set_Ft_design_mat(test1_F)
        test1_D0.set_Gt_transition_mat(test1_G)
        test1_D0.set_Vt_obs_eq_covariance(test1_V)
        test1_D0.set_Wt_state_error_cov(test1_V)
        test1_D0.set_u_no_covariate()

        test1_sim_inst = DLM_simulator(test1_D0, 20220814)
        test1_sim_inst.simulate_data(np.array([0]), np.array([[1]]))
        test1_theta_seq, test1_y_seq = test1_sim_inst.get_theta_y()
        # print(test1_theta_seq)
        # print(test1_y_seq)

        plt.plot(range(100), test1_theta_seq)
        plt.scatter(range(100), test1_y_seq, s=10) #blue dot: obs
        plt.show()

        test1_filter_inst = KalmanFilter(test1_y_seq, test1_D0, np.array([0]), np.array([[1]]))
        test1_filter_inst.run_filter()
        test1_filtered_theta_on_time = test1_filter_inst.theta_on_time
        test1_filtered_theta_one_step_forecast = test1_filter_inst.theta_one_step_forecast
        print(test1_filtered_theta_on_time)
        plt.plot(range(100), test1_theta_seq) #blue: true theta
        plt.plot(range(100), test1_filtered_theta_on_time) #orange: posterior E(theta_t|D_t)
        plt.plot(range(100), test1_filtered_theta_one_step_forecast) #green: prior E(theta_t|D_{t-1})
        plt.scatter(range(100), test1_y_seq, s=10) #blue dot: obs
        plt.show()

    if test2:
        test2_W = [np.array([[1, 0.8],[0.8, 1]]) for _ in range(100)]
        test2_V = [np.array([[0.1, 0],[0, 0.1]]) for _ in range(100)]
        test2_F = [np.array([[1,0],[0,1]]) for _ in range(100)]
        test2_G = [np.array([[0.9, 0], [0, 0.5]]) for _ in range(100)]
        test2_D0 = DLM_D0_container(100)
        test2_D0.set_Ft_design_mat(test2_F)
        test2_D0.set_Gt_transition_mat(test2_G)
        test2_D0.set_Vt_obs_eq_covariance(test2_V)
        test2_D0.set_Wt_state_error_cov(test2_V)
        test2_D0.set_u_no_covariate()

        test2_sim_inst = DLM_simulator(test2_D0, 20220814)
        test2_sim_inst.simulate_data(np.array([0,0]), np.array([[1,0],[0,1]]))
        test2_theta_seq, test2_y_seq = test2_sim_inst.get_theta_y()

        print(test2_theta_seq)
        print(test2_y_seq)
        plt.plot(range(100), [x[0] for x in test2_theta_seq]) #blue: true theta
        plt.scatter(range(100), [x[0] for x in test2_y_seq], s=10) #blue dot: obs
        plt.show()
        # plt.plot(range(100), [x[1] for x in test2_theta_seq])
        # plt.plot(range(100), [x[1] for x in test2_y_seq])
        # plt.show()

        test2_filter_inst = KalmanFilter(test2_y_seq, test2_D0, np.array([0,0]), np.array([[1,0],[0,1]]))
        test2_filter_inst.run_filter()
        test2_filtered_theta_on_time = test2_filter_inst.theta_on_time
        test2_filtered_theta_one_step_forecast = test2_filter_inst.theta_one_step_forecast
        print(test2_filtered_theta_on_time)
        plt.plot(range(100), [x[0] for x in test2_theta_seq]) #blue: true theta
        plt.plot(range(100), [x[0] for x in test2_filtered_theta_on_time]) #orange: posterior E(theta_t|D_t)
        plt.plot(range(100), [x[0] for x in test2_filtered_theta_one_step_forecast]) #green: prior E(theta_t|D_{t-1})
        plt.scatter(range(100), [x[0] for x in test2_y_seq], s=10) #blue dot: obs
        plt.show()
