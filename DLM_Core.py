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


class DLM_full_D0:
    def __init__(self, y_observation, D0: DLM_D0_container, initial_m0_given_D0: np.array, initial_C0_given_D0: np.array):
        self.util_inst = DLM_utility()

        #input
        self.y_observation = self.util_inst.vectorize_seq(y_observation)
        self.y_len = len(y_observation)
        self.D0 = D0
        self.m0 = initial_m0_given_D0
        self.C0 = initial_C0_given_D0

        #result containers
        self.m_posterior_mean = [initial_m0_given_D0]
        self.C_posterior_var = [initial_C0_given_D0]
        self.a_prior_mean = []
        self.R_prior_var = []
        self.f_one_step_forecast_mean = []
        self.Q_one_step_forecast_var = []
        self.A_new_info_scaler = []

        self.ra_reversed_retrospective_a = []
        self.rR_reversed_retrospective_R = []
        self.rB_retrospective_gain_B = []


    def _one_iter(self, t):
        # prior
        Gt = self.D0.G_sys_eq_transition[t-1]
        at = Gt @ self.m_posterior_mean[-1]
        Rt = Gt @ self.C_posterior_var[-1] @ np.transpose(Gt) + self.D0.W_sys_eq_covariance[t-1]

        # one-step forecast
        Ft = self.D0.F_obs_eq_design[t-1]
        ft = np.transpose(Ft) @ at
        Qt = np.transpose(Ft) @ Rt @ Ft + self.D0.V_obs_eq_covariance[t-1]

        # posterior
        At = Rt @ Ft @ np.linalg.inv(Qt)
        et = self.y_observation[t-1] - ft
        mt = at + At @ et
        Ct = Rt - At @ Qt @ np.transpose(At)

        #save
        self.m_posterior_mean.append(mt)
        self.C_posterior_var.append(Ct)
        self.a_prior_mean.append(at)
        self.R_prior_var.append(Rt)
        self.f_one_step_forecast_mean.append(ft)
        self.Q_one_step_forecast_var.append(Qt)
        self.A_new_info_scaler.append(At)

    def run(self):
        for t in range(1, self.y_len+1):
            self._one_iter(t)
        # delete initial value
        self.m_posterior_mean = self.m_posterior_mean[1:]
        self.C_posterior_var = self.C_posterior_var[1:]

    def _retro_one_iter(self, t, k):
        i = t-k
        B_i = self.C_posterior_var[i-1] @ np.transpose(self.D0.G_sys_eq_transition[i]) @ np.linalg.inv(self.R_prior_var[i])
        a_t_k = self.m_posterior_mean[i-1] + B_i @ (self.ra_reversed_retrospective_a[-1] - self.a_prior_mean[i])
        R_t_k = self.C_posterior_var[i-1] + B_i @ (self.rR_reversed_retrospective_R[-1] - self.R_prior_var[i]) @ np.transpose(B_i)

        self.ra_reversed_retrospective_a.append(a_t_k)
        self.rR_reversed_retrospective_R.append(R_t_k)


    def run_retrospective_analysis(self, t_of_given_Dt=None):
        if t_of_given_Dt is None:
            t = self.y_len
        else:
            t = t_of_given_Dt
        
        self.ra_reversed_retrospective_a = [self.m_posterior_mean[t-1]]
        self.rR_reversed_retrospective_R = [self.C_posterior_var[t-1]]
        
        for k in range(1,t):
            self._retro_one_iter(t, k)
        
    
    # == getters ==
    def get_posterior_m_C(self):
        return self.m_posterior_mean, self.C_posterior_var

    def get_prior_a_R(self):
        return self.a_prior_mean, self.R_prior_var

    def get_one_step_forecast_f_Q(self):
        return self.f_one_step_forecast_mean, self.Q_one_step_forecast_var

    def get_retrospective_a_R(self):
        a_smoothed = list(reversed(self.ra_reversed_retrospective_a))
        R_smoothed = list(reversed(self.rR_reversed_retrospective_R))
        return a_smoothed, R_smoothed


class DLM_univariate_y_without_V_in_D0:
    #chapter 4.5 of West
    #conjugate analysis. when V is unknown, y is univariate
    def __init__(self, y_observation, D0_having_F_G_Wst: DLM_D0_container,
                initial_m0_given_D0: np.array, initial_C0st_given_D0: np.array, n0_given_D0:float, S0_given_D0:float):
        self.util_inst = DLM_utility()

        self.y_observation = self.util_inst.vectorize_seq(y_observation)
        self.y_len = len(y_observation)
        self.D0 = D0_having_F_G_Wst
        self.m0 = initial_m0_given_D0
        self.C0st = initial_C0st_given_D0
        self.n0 = n0_given_D0
        self.S0 = S0_given_D0

        #result containers
        self.m_posterior_mean = [initial_m0_given_D0]
        self.Cst_posterior_var = [initial_C0st_given_D0]
        self.C_posterior_scale = [S0_given_D0*initial_C0st_given_D0]
        self.a_prior_mean = []
        self.Rst_prior_var = []
        self.R_prior_scale = []
        self.f_one_step_forecast_mean = []
        self.Qst_one_step_forecast_var = []
        self.Q_one_step_forecast_scale = []
        self.A_new_info_scaler = []
        self.n_precision_shape = [n0_given_D0] #shape: {n_t}/2 at t
        self.S_precision_rate = [S0_given_D0] #rate: {n_t}{S_t}/2 at t
        self.ra_reversed_retrospective_a = []
        self.rR_reversed_retrospective_R = []

    def _one_iter(self, t):
        #conditional on V
        ##prior
        Gt = self.D0.G_sys_eq_transition[t-1]
        at = Gt @ self.m_posterior_mean[-1]
        Rst_t = Gt @ self.Cst_posterior_var[-1] @ np.transpose(Gt) + self.D0.W_sys_eq_covariance[t-1]
        ##one_step_forecast
        Ft = self.D0.F_obs_eq_design[t-1]
        ft = np.transpose(Ft) @ at
        Qst_t = 1 + np.transpose(Ft) @ Rst_t @ Ft
        ##posterior
        et = self.y_observation[t-1] - ft
        At = Rst_t @ Ft @ np.linalg.inv(Qst_t)
        mt = at + At @ et
        Cst_t = Rst_t - At @ np.transpose(At) @ Qst_t

        #precision
        nt = self.n_precision_shape[-1] + 1
        S_t1 = self.S_precision_rate[-1]
        St = float(S_t1*(nt-1)/nt + et @ et / (nt * Qst_t))

        #unconditional on V
        Ct = St * Cst_t
        Rt = S_t1 * Rst_t
        Qt = S_t1 * Qst_t

        #save
        self.m_posterior_mean.append(mt)
        self.Cst_posterior_var.append(Cst_t)
        self.C_posterior_scale.append(Ct)
        self.a_prior_mean.append(at)
        self.Rst_prior_var.append(Rst_t)
        self.R_prior_scale.append(Rt)
        self.f_one_step_forecast_mean.append(ft)
        self.Qst_one_step_forecast_var.append(Qst_t)
        self.Q_one_step_forecast_scale.append(Qt)
        self.A_new_info_scaler.append(At)
        self.n_precision_shape.append(nt)
        self.S_precision_rate.append(St)
    
    def run(self):
        for t in range(1, self.y_len+1):
            self._one_iter(t)
        # delete initial value
        self.m_posterior_mean = self.m_posterior_mean[1:]
        self.Cst_posterior_var = self.Cst_posterior_var[1:]
        self.C_posterior_scale = self.C_posterior_scale[1:]
        self.n_precision_shape = self.n_precision_shape[1:]
        self.S_precision_rate = self.S_precision_rate[1:]

    #retrospective analysis: should I use C, R? or C_star, R_star?
    #now, use C,R
    def _retro_one_iter(self, t, k):
        i = t-k
        B_i = self.C_posterior_scale[i-1] @ np.transpose(self.D0.G_sys_eq_transition[i]) @ np.linalg.inv(self.R_prior_scale[i])
        a_t_k = self.m_posterior_mean[i-1] + B_i @ (self.ra_reversed_retrospective_a[-1] - self.a_prior_mean[i])
        R_t_k = self.C_posterior_scale[i-1] + B_i @ (self.rR_reversed_retrospective_R[-1] - self.R_prior_scale[i]) @ np.transpose(B_i)

        self.ra_reversed_retrospective_a.append(a_t_k)
        self.rR_reversed_retrospective_R.append(R_t_k)


    def run_retrospective_analysis(self, t_of_given_Dt=None):
        if t_of_given_Dt is None:
            t = self.y_len
        else:
            t = t_of_given_Dt
        
        self.ra_reversed_retrospective_a = [self.m_posterior_mean[t-1]]
        self.rR_reversed_retrospective_R = [self.C_posterior_scale[t-1]]
        
        for k in range(1,t):
            self._retro_one_iter(t, k)

    # == getters ==
    def get_posterior_m_C(self):
        return self.m_posterior_mean, self.C_posterior_scale

    def get_prior_a_R(self):
        return self.a_prior_mean, self.R_prior_scale

    def get_one_step_forecast_f_Q(self):
        return self.f_one_step_forecast_mean, self.Q_one_step_forecast_scale

    
    def get_retrospective_a_R(self):
        a_smoothed = list(reversed(self.ra_reversed_retrospective_a))
        R_smoothed = list(reversed(self.rR_reversed_retrospective_R))
        return a_smoothed, R_smoothed


if __name__=="__main__":
    import matplotlib.pyplot as plt
    test1 = False
    test2 = True

    if test1:
        test1_W = [np.array([[1]]) for _ in range(100)]
        test1_V = [np.array([[0.5]]) for _ in range(100)]
        test1_F = [np.array([[0.1]]) for _ in range(100)]
        test1_G = [np.array([[1]]) for _ in range(100)]
        test1_D0 = DLM_D0_container(100)
        test1_D0.set_Ft_design_mat(test1_F)
        test1_D0.set_Gt_transition_mat(test1_G)
        test1_D0.set_Vt_obs_eq_covariance(test1_V)
        test1_D0.set_Wt_state_error_cov(test1_W)
        test1_D0.set_u_no_covariate()

        test1_sim_inst = DLM_simulator(test1_D0, 20220815)
        test1_sim_inst.simulate_data(np.array([0]), np.array([[1]]))
        test1_theta_seq, test1_y_seq = test1_sim_inst.get_theta_y()
        # print(test1_theta_seq)
        # print(test1_y_seq)

        plt.plot(range(100), test1_theta_seq)
        plt.scatter(range(100), test1_y_seq, s=10) #blue dot: obs
        plt.show()

        test1_DLM_inst = DLM_full_D0(test1_y_seq, test1_D0, np.array([0]), np.array([[1]]))
        test1_DLM_inst.run()
        test1_posterior_m = test1_DLM_inst.m_posterior_mean
        test1_one_step_forecast_f = test1_DLM_inst.f_one_step_forecast_mean
        test1_one_step_forecast_var = test1_DLM_inst.Q_one_step_forecast_var
        # print(test1_filtered_theta_on_time)
        plt.plot(range(100), test1_theta_seq) #blue: true theta
        plt.plot(range(100), test1_posterior_m) #orange: posterior E(theta_t|D_t)
        plt.plot(range(100), test1_one_step_forecast_f) #green: one-step forecast E(Y_t|D_{t-1})
        plt.scatter(range(100), test1_y_seq, s=10) #blue dot: obs
        z95 = 1.644854
        cred_interval_upper = [x[0] + z95*np.sqrt(v[0][0]) for x, v in zip(test1_one_step_forecast_f, test1_one_step_forecast_var)]
        cred_interval_lower = [x[0] - z95*np.sqrt(v[0][0]) for x, v in zip(test1_one_step_forecast_f, test1_one_step_forecast_var)]
        plt.plot(range(100), cred_interval_upper, color="grey") #one-step forecast (Y_t|D_{t-1}) 90% credible interval
        plt.plot(range(100), cred_interval_lower, color="grey") #one-step forecast (Y_t|D_{t-1}) 90% credible interval
        plt.show()

        test1_DLM_inst.run_retrospective_analysis(50)
        test1_retro_a_at_50, test1_retro_R_at_50 = test1_DLM_inst.get_retrospective_a_R()
        test1_DLM_inst.run_retrospective_analysis()
        test1_retro_a_at_100, test1_retro_R_at_100 = test1_DLM_inst.get_retrospective_a_R()
        plt.plot(range(100), test1_theta_seq) #blue: true theta
        plt.scatter(range(100), test1_y_seq, s=10) #blue dot: obs
        plt.plot(range(50), test1_retro_a_at_50) #orange: smoothed at t=50
        plt.plot(range(100), test1_retro_a_at_100) #green: smoothed at t=100(end pt)
        plt.show()

    
    if test2:
        test2_true_W = [np.array([[0.5]]) for _ in range(100)]
        test2_true_V = [np.array([[1]]) for _ in range(100)]
        test2_true_F = [np.array([[1]]) for _ in range(100)]
        test2_true_G = [np.array([[1]]) for _ in range(100)]
        test2_true_D0 = DLM_D0_container(100)
        test2_true_D0.set_Ft_design_mat(test2_true_F)
        test2_true_D0.set_Gt_transition_mat(test2_true_G)
        test2_true_D0.set_Vt_obs_eq_covariance(test2_true_V)
        test2_true_D0.set_Wt_state_error_cov(test2_true_W)
        test2_true_D0.set_u_no_covariate()

        test2_Wst = [np.array([[0.5]]) for _ in range(100)] #true:0.5
        test2_F = [np.array([[1]]) for _ in range(100)] #true:1
        test2_G = [np.array([[1]]) for _ in range(100)] #true:1
        test2_D0 = DLM_D0_container(100)
        test2_D0.set_Ft_design_mat(test2_F)
        test2_D0.set_Gt_transition_mat(test2_G)
        test2_D0.set_Wt_state_error_cov(test2_Wst)
        test2_D0.set_u_no_covariate()

        test2_sim_inst = DLM_simulator(test2_true_D0, 20220815)
        test2_sim_inst.simulate_data(np.array([0]), np.array([[1]]))
        test2_theta_seq, test2_y_seq = test2_sim_inst.get_theta_y()
        # print(test2_theta_seq)
        # print(test2_y_seq)

        plt.plot(range(100), test2_theta_seq)
        plt.scatter(range(100), test2_y_seq, s=10) #blue dot: obs
        plt.show()

        test2_DLM_inst = DLM_univariate_y_without_V_in_D0(
                test2_y_seq, test2_D0,
                np.array([0]), np.array([[1]]), 1, 0.1)
        test2_DLM_inst.run()
        test2_posterior_m = test2_DLM_inst.m_posterior_mean
        test2_one_step_forecast_f = test2_DLM_inst.f_one_step_forecast_mean
        test2_one_step_forecast_var = test2_DLM_inst.Q_one_step_forecast_scale
        # print(test1_filtered_theta_on_time)
        plt.plot(range(100), test2_theta_seq) #blue: true theta
        plt.plot(range(100), test2_posterior_m) #orange: posterior E(theta_t|D_t)
        plt.plot(range(100), test2_one_step_forecast_f) #green: one-step forecast E(Y_t|D_{t-1})
        plt.scatter(range(100), test2_y_seq, s=10) #blue dot: obs
        
        z95 = 1.644854 #comment: for exact Cred.interval, we need to use T_{nt} distribution. but...
        cred_interval_upper = [x[0] + z95*np.sqrt(v[0][0]) for x, v in zip(test2_one_step_forecast_f, test2_one_step_forecast_var)]
        cred_interval_lower = [x[0] - z95*np.sqrt(v[0][0]) for x, v in zip(test2_one_step_forecast_f, test2_one_step_forecast_var)]
        plt.plot(range(100), cred_interval_upper, color="grey") #one-step forecast (Y_t|D_{t-1}) 90% credible interval
        plt.plot(range(100), cred_interval_lower, color="grey") #one-step forecast (Y_t|D_{t-1}) 90% credible interval
        plt.show()

        test2_DLM_inst.run_retrospective_analysis(50)
        test2_retro_a_at_50, test2_retro_R_at_50 = test2_DLM_inst.get_retrospective_a_R()
        test2_DLM_inst.run_retrospective_analysis()
        test2_retro_a_at_100, test2_retro_R_at_100 = test2_DLM_inst.get_retrospective_a_R()
        plt.plot(range(100), test2_theta_seq) #blue: true theta
        plt.scatter(range(100), test2_y_seq, s=10) #blue dot: obs
        plt.plot(range(50), test2_retro_a_at_50) #orange: smoothed at t=50
        plt.plot(range(100), test2_retro_a_at_100) #green: smoothed at t=100(end pt)
        plt.show()
