#! /usr/bin/env python
##################################################
# Author: Noemie Jaquier, 2019
# License: MIT
# Contact: noemie.jaquier@idiap.ch
##################################################
import os
# from ..forward_mode.path_planning.utils import *
import numpy as np
import GPy
import matplotlib.pyplot as plt
from scipy.io import loadmat
from forward_mode.utils.gmr import Gmr
from forward_mode.utils.gmr import plot_gmm
from forward_mode.utils.gp_coregionalize_with_mean_regression import GPCoregionalizedWithMeanRegression
from forward_mode.utils.gmr_mean_mapping import GmrMeanMapping
from forward_mode.utils.gmr_kernels import Gmr_based_kernel


def cope_real_word_path(
    root_path=None,  
    file_name='mu',  
    stroke_index=1,  
    epi_times=1,  
    delimiter=' ',  
    skiprows=1,  
):
    Length = [0.30, 0.150, 0.25, 0.125]
    L_1 = Length[0]
    L_2 = Length[2]
    
    action_dim = 2
    Ts = 0.001
    
    # writing space
    WIDTH = 0.360
    HEIGHT = 0.360
    
    linewidth = 3.0
    x_list = []
    y_list = []
    for epi_index in range(epi_times):
        angle_list = np.loadtxt(
            root_path + file_name + str(stroke_index) + '_' + str(epi_index) + '.txt',
            delimiter=delimiter,
            skiprows=skiprows
            )

        # angle_list : desired and real-time
        angle_list_1_e = angle_list[:, 0]
        angle_list_2_e = angle_list[:, 3]
        angle_list_1_t = angle_list[:, 1]
        angle_list_2_t = angle_list[:, 4]

        x_e = L_1 * np.cos(angle_list_1_e) + L_2 * np.cos(angle_list_1_e + angle_list_2_e)
        y_e = L_1 * np.sin(angle_list_1_e) + L_2 * np.sin(angle_list_1_e + angle_list_2_e)

        x_t = L_1 * np.cos(angle_list_1_t) + L_2 * np.cos(angle_list_1_t + angle_list_2_t)
        y_t = L_1 * np.sin(angle_list_1_t) + L_2 * np.sin(angle_list_1_t + angle_list_2_t)

        x_list.append(x_t)
        y_list.append(y_t)

    return x_list, y_list


def scale_translate_process_main(
    X_list, Y_list,  
    scale_factor=np.array([1.0, 1.0]),   
    trans_value=np.array([0.0, 0.0])   
):
    """
        process original data
    """
    X_list = X_list - trans_value[0]
    Y_list = Y_list - trans_value[1]
    
    X_list = X_list * scale_factor[0]
    Y_list = Y_list * scale_factor[1]
    
    return X_list, Y_list


def word_process_main(
    write_name='ren_10',  
    stroke_index=1,  
    file_fig_name='./data/predicted_images/',  
    re_sample_index=20,
    epi_times=5,
    dt=0.01
):
    input_dim = 1 
    output_dim = 2 
    in_idx = [0] 
    out_idx = [1, 2] 

    nb_data_sup = 0  

    # =====================================================
    ############# process data before prediction ##########
    # =====================================================
    x_list, y_list = cope_real_word_path(
        root_path='./data/font_data/' + write_name + '/',
        file_name='real_angle_list_',
        stroke_index=stroke_index,
        epi_times=epi_times,
        delimiter=',',
        skiprows=1 
    )
    
    # scale data
    x_list, y_list = \
        scale_translate_process_main(
            x_list, y_list,
            scale_factor=np.array([100, 100]),
            trans_value=np.array([0.3, 0.0])  
        )  
    
    print("x_list :", x_list.shape, len(x_list))
    
    folder_name = file_fig_name + write_name
    if os.path.exists(folder_name):
        pass 
    else: 
        os.makedirs(folder_name)

    x_down_list = []  
    y_down_list = []  
    for i in range(len(x_list)):
        x_down_list.append(x_list[i][::re_sample_index])   
        y_down_list.append(y_list[i][::re_sample_index])   
    
    print('x_list shape :', np.array(x_down_list).shape, 'y list shape :', np.array(y_down_list).shape)
    
    # demos = np.zeros_like(x_down_list[0])
    nb_data = x_down_list[0].shape[0]

    # Create time data
    demos_t = [np.arange(x_down_list[i].shape[0])[:, None] for i in range(epi_times)]
    # print("demos_t :", demos_t)

    # Stack time and position data
    demos_tx = [np.hstack([demos_t[i] * dt, x_down_list[i][:, None], y_down_list[i][:, None]]) for i in
                range(epi_times)]
    # print("demos_tx :", demos_tx)

    # Stack demos
    demos_np = demos_tx[0]  
    for i in range(1, epi_times):  
        demos_np = np.vstack([demos_np, demos_tx[i]])

    X = demos_np[:, 0][:, None]   
    Y = demos_np[:, 1:]   
    
    # Train data for GPR
    X_list = [np.hstack((X, X)) for i in range(output_dim)]   
    Y_list = [Y[:, i][:, None] for i in range(output_dim)]   

    Xt = dt * np.arange(nb_data + nb_data_sup)[:, None]
    print("X shape :", X.shape, "Y shape :", Y.shape, "Xt :", Xt.shape, "demos_np :", demos_np.shape)
    print("X_list :", np.array(X_list).shape, "Y_list :", np.array(Y_list).shape)
    # word : yi
    # X_obs = np.array([0.25, 0.60, 1.3, 1.75])[:, None]  
    # x_list_1 = np.array([0.32, 0.315, 0.315, 0.35])
    # y_list_1 = np.array([-0.11, -0.03, 0.05, 0.14])
    
    # word ren stroke : 0
    # X_obs = np.array([0.1, 0.65, 1.65])[:, None]
    # x_list_1 = np.array([0.25, 0.3, 0.42])
    # y_list_1 = np.array([-0.02, -0.05, -0.19])
    # Y_obs = np.array([[0.32, -0.15], [0.3, -0.03], [0.35, 0.16]])
    
    # word ren stroke : 1
    # X_obs = np.array([0.2, 1.0])[:, None]
    # x_list_1 = np.array([0.32, 0.40])
    # y_list_1 = np.array([0.02, 0.11])
    
    # word mu stroke : 0
    # X_obs = np.array([0.3, 1.0])[:, None]
    # x_list_1 = np.array([0.27, 0.275])
    # y_list_1 = np.array([-0.06, 0.07])
    
    # # word mu stroke : 1
    # X_obs = np.array([0.15, 1.0])[:, None]
    # x_list_1 = np.array([0.23, 0.35])
    # y_list_1 = np.array([-0.01, 0.01])
    
    # # word mu stroke : 2
    # X_obs = np.array([0.05, 1.1])[:, None]
    # x_list_1 = np.array([0.3, 0.38])
    # y_list_1 = np.array([-0.03, -0.15])
    
    # # word mu stroke : 3
    X_obs = np.array([0.05, 0.9, 1.3])[:, None]  
    x_list_1 = np.array([0.3, 0.37, 0.38])  
    y_list_1 = np.array([-0.01, 0.09, 0.15])  
    
    x_obs_1, y_obs_1 = \
        scale_translate_process_main(
            x_list_1.copy(), y_list_1.copy(),  
            scale_factor=np.array([100, 100]),  
            trans_value=np.array([0.3, 0.0])  
        )
    Y_obs = np.hstack((x_obs_1[:, None], y_obs_1[:, None])).reshape(-1, 2)
    # print("x_obs_1", x_obs_1, "Y_obs :", Y_obs)
    X_obs_list = [np.hstack((X_obs, X_obs)) for i in range(output_dim)]
    Y_obs_list = [Y_obs[:, i][:, None] for i in range(output_dim)]
    
    return X_list, Y_list, X, Y, X_obs_list, Y_obs_list, X_obs, Y_obs, Xt, epi_times, demos_np, nb_data


def ori_word_process():
    # GMR-based GPR on 2D trajectories with time as input
    file_name = './forward_mode'
    datapath = file_name + '/data/2Dletters/'
    letter = 'B'  # choose a letter in the alphabet
    data = loadmat(datapath + '%s.mat' % letter)
    demos = [d['pos'][0][0].T for d in data['demos'][0]]
    
    # Parameters
    nb_data = demos[0].shape[0]
    print("nb_data :", nb_data)
    nb_data_sup = 0
    nb_samples = 5
    dt = 0.01
    input_dim = 1
    output_dim = 2
    
    # Create time data
    demos_t = [np.arange(demos[i].shape[0])[:, None] + 1 for i in range(nb_samples)]
    
    # Stack time and position data
    demos_tx = [np.hstack([demos_t[i] * dt, demos[i]]) for i in range(nb_samples)]
    print("max_time :", np.max(demos_tx))
    
    # Stack demos
    demos_np = demos_tx[0]
    for i in range(1, nb_samples):
        demos_np = np.vstack([demos_np, demos_tx[i]])
    
    X = demos_np[:, 0][:, None]
    Y = demos_np[:, 1:]
    
    # Train data for GPR
    X_list = [np.hstack((X, X)) for i in range(output_dim)]
    Y_list = [Y[:, i][:, None] for i in range(output_dim)]
    
    # Define via-points (new set of observations)
    X_obs = np.array([0.0, 1., 1.9])[:, None]
    Y_obs = np.array([[-12.5, -11.5], [-0.5, -1.5], [-14.0, -7.5]])
    
    # X_obs = np.array([0.4, 0.5, 1., 1.4])[:, None]
    # Y_obs = np.array([[-7.0, 7.0], [-5, 8.0], [-0.5, -1.5], [3.0, -10.]])
    
    X_obs_list = [np.hstack((X_obs, X_obs)) for i in range(output_dim)]
    Y_obs_list = [Y_obs[:, i][:, None] for i in range(output_dim)]
    
    # Test data
    # Xt = dt * np.arange(demos[0].shape[0] + nb_data_sup)[:, None]
    Xt = dt * np.arange(nb_data + nb_data_sup)[:, None]
    print("Xt :", Xt.shape)
    
    return X_list, Y_list, X, Y, X_obs_list, Y_obs_list, X_obs, Y_obs, Xt, nb_samples, demos_np, nb_data


if __name__ == '__main__':
    # ====================== data processing ======================
    write_name = 'mu' 
    # write_name = 'B' 
    stroke_index = 3 
    file_fig_name = './data/predicted_images/'  
    re_sample_index = 20  
    epi_times = 5  
    dt = 0.01 
    
    # word path from our dataset
    X_list, Y_list, X, Y, X_obs_list, Y_obs_list, X_obs, Y_obs, Xt, nb_samples, demos_np, nb_data = \
        word_process_main(
            write_name=write_name, 
            stroke_index=stroke_index, 
            file_fig_name=file_fig_name, 
            re_sample_index=re_sample_index, 
            epi_times=epi_times, 
            dt=dt
    )
    
    # load data from letter B
    # X_list, Y_list, X, Y, X_obs_list, Y_obs_list, X_obs, Y_obs, Xt, nb_samples, demos_np, nb_data = \
    #     ori_word_process()

    # ===================== training parameters ===================
    input_dim = 1
    output_dim = 2
    in_idx = [0]
    out_idx = [1, 2]
    nb_states = 6

    nb_prior_samples = 10
    nb_posterior_samples = 5 

    # ------------------------ plot figure ------------------------
    font_size = 25

    # ------------------------ Implementation ---------------------
    # GMM
    gmr_model = Gmr(nb_states=nb_states, nb_dim=input_dim + output_dim, in_idx=in_idx, out_idx=out_idx)
    gmr_model.init_params_kbins(demos_np.T, nb_samples=nb_samples)
    gmr_model.gmm_em(demos_np.T)

    # GMR prediction
    mu_gmr = []
    sigma_gmr = []
    for i in range(Xt.shape[0]):
        mu_gmr_tmp, sigma_gmr_tmp, H_tmp = gmr_model.gmr_predict(Xt[i])
        mu_gmr.append(mu_gmr_tmp)
        sigma_gmr.append(sigma_gmr_tmp)

    mu_gmr = np.array(mu_gmr)
    sigma_gmr = np.array(sigma_gmr)

    plt.figure(figsize=(5, 5))
    for p in range(nb_samples):
        plt.plot(Y[p * nb_data:(p + 1) * nb_data, 0], Y[p * nb_data:(p + 1) * nb_data, 1], color=[.7, .7, .7])
        plt.scatter(Y[p * nb_data, 0], Y[p * nb_data, 1], color=[.7, .7, .7], marker='X', s=50)
    
    plt.plot(mu_gmr[:, 0], mu_gmr[:, 1], color=[0.20, 0.54, 0.93], linewidth=3)
    plt.scatter(mu_gmr[0, 0], mu_gmr[0, 1], color=[0.20, 0.54, 0.93], marker='X', s=50)
    plot_gmm(mu_gmr, sigma_gmr, alpha=0.05, color=[0.20, 0.54, 0.93])
    plt.scatter(Y_obs[:, 0], Y_obs[:, 1], color=[0, 0, 0], zorder=60, s=100)

    axes = plt.gca()
    # axes.set_xlim([0.1, 0.45])
    # axes.set_ylim([-0.25, 0.25])
    # axes.set_xlim([-10, 10])
    axes.set_xlim([-20, 20])
    axes.set_ylim([-20, 20])
    # axes.set_ylim([-10, 10])
    # axes.set_xlim([-15, 15])
    # axes.set_ylim([-15, 15])
    plt.xlabel('$x_1$', fontsize=font_size)
    plt.ylabel('$x_2$', fontsize=font_size)
    plt.locator_params(nbins=3)
    plt.tick_params(labelsize=font_size)
    plt.tight_layout()
    # plt.savefig(file_fig_name + '/' + write_name + '/' + 'GMR_' + write_name + '_stroke_' + str(stroke_index) + '.pdf')

    plt.show()

    # # plt.figure(figsize=(5, 4))
    # # for p in range(nb_samples):
    # #     plt.plot(Xt[:nb_data, 0], Y[p * nb_data:(p + 1) * nb_data, 0], color=[.7, .7, .7])
    # # plt.plot(Xt[:, 0], mu_gmr[:, 0], color=[0.20, 0.54, 0.93], linewidth=3)
    # # # plt.plot(Xt[:, 0], prior_traj_sample_1, color=[0.20, 0.54, 0.93], linewidth=3)
    # # miny = mu_gmr[:, 0] - np.sqrt(sigma_gmr[:, 0, 0])
    # # maxy = mu_gmr[:, 0] + np.sqrt(sigma_gmr[:, 0, 0])
    # # plt.fill_between(Xt[:, 0], miny, maxy, color=[0.20, 0.54, 0.93], alpha=0.3)
    # # prior_traj = []
    # #
    # # axes = plt.gca()
    # # # axes.set_ylim([-14., 14.])
    # # plt.xlabel('$t$', fontsize=30)
    # # plt.ylabel('$x_1(m)$', fontsize=30)
    # # plt.tick_params(labelsize=20)
    # # plt.tight_layout()
    # # # plt.savefig(file_name + '/figures/GMR_B01.png')


    # # ========================= GPR ==========================
    # # ========================================================
    # # Define GPR likelihood and kernels ::: original : 0.01
    # nb_data_test = Xt.shape[0]
    # Xtest, _, output_index = GPy.util.multioutput.build_XY([np.hstack((Xt, Xt)) for i in range(output_dim)])

    # likelihoods_list = [GPy.likelihoods.Gaussian(name="Gaussian_noise_%s" %j, variance=1) for j in range(output_dim)]
    # # kernel_list = [GPy.kern.RBF(1, variance=0.001, lengthscale=0.5) for i in range(gmr_model.nb_states)]
    # # kernel_list = [GPy.kern.RBF(1, variance=5, lengthscale=2) for i in range(gmr_model.nb_states)]
    # kernel_list = [GPy.kern.Matern52(1, variance=5., lengthscale=0.5) for i in range(gmr_model.nb_states)]

    # # Fix variance of kernels
    # for kernel in kernel_list:
    #     kernel.variance.fix(1.0)  # 1.0www
    #     kernel.lengthscale.constrain_bounded(1.5, 10.)

    # # Bound noise parameters
    # for likelihood in likelihoods_list:
    #     likelihood.variance.constrain_bounded(0.001, 0.05)

    # # GPR model
    # K = Gmr_based_kernel(gmr_model=gmr_model, kernel_list=kernel_list)
    # mf = GmrMeanMapping(2 * input_dim + 1, 1, gmr_model)
    # m = GPCoregionalizedWithMeanRegression(
    #     X_list, Y_list,
    #     kernel=K,
    #     likelihoods_list=likelihoods_list,
    #     mean_function=mf
    # )

    # # Parameters optimization
    # m.optimize('bfgs', max_iters=200, messages=True)

    # # Print model parameters
    # print(m)

    # # GPR prior (no observations)
    # prior_traj = []
    # prior_mean = mf.f(Xtest)[:, 0]
    # prior_kernel = m.kern.K(Xtest)
    # for i in range(nb_prior_samples):
    #     print("prior_kernel :", prior_kernel.shape)
    #     prior_traj_tmp = np.random.multivariate_normal(prior_mean, prior_kernel)
    #     print("prior_traj_tmp :", prior_traj_tmp.shape)
    #     prior_traj.append(np.reshape(prior_traj_tmp, (output_dim, -1)))
    # prior_kernel_tmp = np.zeros((nb_data_test, nb_data_test, output_dim * output_dim))
    # for i in range(output_dim):
    #     for j in range(output_dim):
    #         prior_kernel_tmp[:, :, i * output_dim + j] = prior_kernel[i * nb_data_test:(i + 1) * nb_data_test, j * nb_data_test:(j + 1) * nb_data_test]
    # prior_kernel_rshp = np.zeros((nb_data_test, output_dim, output_dim))
    # for i in range(nb_data_test):
    #     prior_kernel_rshp[i] = np.reshape(prior_kernel_tmp[i, i, :], (output_dim, output_dim))

    # # GPR posterior -> new points observed (the training points are discarded as they are "included" in the GMM)
    # m_obs = \
    #     GPCoregionalizedWithMeanRegression(
    #         X_obs_list, Y_obs_list, kernel=K,
    #         likelihoods_list=likelihoods_list,
    #         mean_function=mf
    #     )
    # mu_posterior_tmp = \
    #     m_obs.posterior_samples_f(
    #         Xtest, full_cov=True, size=nb_posterior_samples
    #     )

    # mu_posterior = []
    # for i in range(nb_posterior_samples):
    #     mu_posterior.append(np.reshape(mu_posterior_tmp[:, 0, i], (output_dim, -1)))

    # # GPR prediction
    # mu_gp, sigma_gp = m_obs.predict(Xtest, full_cov=True, Y_metadata={'output_index': output_index})
    # mu_gp_rshp = np.reshape(mu_gp, (output_dim, -1)).T
    # sigma_gp_tmp = np.zeros((nb_data_test, nb_data_test, output_dim * output_dim))
    # for i in range(output_dim):
    #     for j in range(output_dim):
    #         sigma_gp_tmp[:, :, i * output_dim + j] = sigma_gp[i * nb_data_test:(i + 1) * nb_data_test, j * nb_data_test:(j + 1) * nb_data_test]
    # sigma_gp_rshp = np.zeros((nb_data_test, output_dim, output_dim))
    # for i in range(nb_data_test):
    #     sigma_gp_rshp[i] = np.reshape(sigma_gp_tmp[i, i, :], (output_dim, output_dim))
    # print("sigma_gp_rshp", sigma_gp_rshp.shape)

    # # Final plots
    # # GMM
    # # plt.figure(figsize=(5, 5))
    # # for p in range(nb_samples):
    # # 	plt.plot(Y[p * nb_data:(p + 1) * nb_data, 0], Y[p * nb_data:(p + 1) * nb_data, 1], color=[.7, .7, .7])
    # # 	plt.plot(Y[p * nb_data, 0], Y[p * nb_data, 1], color=[.7, .7, .7], marker='o')
    # # plot_gmm(np.array(gmr_model.mu)[:, 1:], np.array(gmr_model.sigma)[:, 1:, 1:], alpha=0.6, color=[0.1, 0.34, 0.73])
    # # plt.savefig(file_name + '/figures/GMRbGP_B_gmm.png')

    # # Priors
    # plt.figure(figsize=(5, 5))
    # plt.plot(mu_gmr[:, 0], mu_gmr[:, 1], color=[0.20, 0.54, 0.93], linewidth=3.)
    # plt.scatter(mu_gmr[0, 0], mu_gmr[0, 1], color=[0.20, 0.54, 0.93], marker='X', s=80)
    # plot_gmm(mu_gmr, prior_kernel_rshp, alpha=0.05, color=[0.64, 0.27, 0.73])

    # for i in range(nb_prior_samples):
    #     plt.plot(prior_traj[i][0], prior_traj[i][1], color=[0.64, 0.27, 0.73], linewidth=1.)
    #     plt.scatter(prior_traj[i][0, 0], prior_traj[i][1, 0], color=[0.64, 0.27, 0.73], marker='X', s=80)

    # axes = plt.gca()
    # # axes.set_xlim([-10, 10])
    # axes.set_xlim([-20, 20])
    # axes.set_ylim([-20, 20])
    # # axes.set_ylim([-10, 10])
    # # axes.set_xlim([-15., 15.])
    # # axes.set_ylim([-15., 15.])
    # plt.xlabel('$x_1$', fontsize=font_size)
    # plt.ylabel('$x_2$', fontsize=font_size)
    # plt.locator_params(nbins=3)
    # plt.tick_params(labelsize=font_size)
    # plt.tight_layout()
    # plt.savefig(file_fig_name + '/' + write_name + '/' + 'GMRbGP_' + write_name + '_stroke_' + str(stroke_index) + '_priors.pdf')

    # # plt.figure(figsize=(5, 4))
    # # plt.plot(Xt[:, 0], mu_gmr[:, 0], color=[0.20, 0.54, 0.93], linewidth=3)
    # # miny = mu_gmr[:, 0] - np.sqrt(prior_kernel_rshp[:, 0, 0])
    # # maxy = mu_gmr[:, 0] + np.sqrt(prior_kernel_rshp[:, 0, 0])
    # # plt.fill_between(Xt[:, 0], miny, maxy, color=[0.64, 0.27, 0.73], alpha=0.3)
    # # for i in range(nb_prior_samples):
    # #     plt.plot(Xt[:, 0], prior_traj[i][0], color=[0.64, 0.27, 0.73], linewidth=1.)
    # # axes = plt.gca()
    # # axes.set_ylim([-17., 17.])
    # # plt.xlabel('$t$', fontsize=30)
    # # plt.ylabel('$y_1$', fontsize=30)
    # # plt.tick_params(labelsize=20)
    # # plt.tight_layout()
    # # plt.savefig(file_name + '/figures/GMRbGP_B_priors01_datasup.png')

    # # plt.figure(figsize=(5, 4))
    # # for p in range(nb_samples):
    # #     plt.plot(Xt[:nb_data, 0], Y[p * nb_data:(p + 1) * nb_data, 1], color=[.7, .7, .7])
    # # plt.plot(Xt[:, 0], mu_gmr[:, 1], color=[0.20, 0.54, 0.93], linewidth=3.)
    # # miny = mu_gmr[:, 1] - np.sqrt(prior_kernel_rshp[:, 1, 1])
    # # maxy = mu_gmr[:, 1] + np.sqrt(prior_kernel_rshp[:, 1, 1])
    # # plt.fill_between(Xt[:, 0], miny, maxy, color=[0.64, 0.27, 0.73], alpha=0.3)
    # # for i in range(nb_prior_samples):
    # #     plt.plot(Xt[:, 0], prior_traj[i][1], color=[0.64, 0.27, 0.73], linewidth=1.)
    # # axes = plt.gca()
    # # axes.set_ylim([-17., 17.])
    # # plt.xlabel('$t$', fontsize=30)
    # # plt.ylabel('$y_2$', fontsize=30)
    # # plt.tick_params(labelsize=20)
    # # plt.tight_layout()
    # # plt.savefig(file_name + '/figures/GMRbGP_B_priors02_datasup.png')

    # # Posterior
    # plt.figure(figsize=(5, 5))
    # plt.plot(mu_gmr[:, 0], mu_gmr[:, 1], color=[0.20, 0.54, 0.93], linewidth=3.)
    # plot_gmm(mu_gp_rshp, sigma_gp_rshp, alpha=0.05, color=[0.83, 0.06, 0.06])
    # for i in range(nb_posterior_samples):
    #     plt.plot(mu_posterior[i][0], mu_posterior[i][1], color=[0.64, 0., 0.65], linewidth=1.5)
    #     plt.scatter(mu_posterior[i][0, 0], mu_posterior[i][1, 0], color=[0.64, 0., 0.65], marker='X', s=80)

    # plt.plot(mu_gp_rshp[:, 0], mu_gp_rshp[:, 1], color=[0.83, 0.06, 0.06], linewidth=3.)
    # plt.scatter(mu_gp_rshp[0, 0], mu_gp_rshp[0, 1], color=[0.83, 0.06, 0.06], marker='X', s=80)
    # plt.scatter(Y_obs[:, 0], Y_obs[:, 1], color=[0, 0, 0], zorder=60, s=100)

    # axes = plt.gca()
    # axes.set_xlim([-20, 20])
    # # axes.set_xlim([-10, 10])
    # axes.set_ylim([-20, 20])
    # # axes.set_ylim([-10, 10])
    # # axes.set_xlim([-15, 15])
    # # axes.set_ylim([-15, 15])
    # plt.xlabel('$x_1$', fontsize=font_size)
    # plt.ylabel('$x_2$', fontsize=font_size)
    # plt.locator_params(nbins=3)
    # plt.tick_params(labelsize=font_size)
    # plt.tight_layout()
    # # plt.savefig(file_name + '/figures/GMRbGP_B_posterior_datasup.png')
    # plt.savefig(file_fig_name + '/' + write_name + '/' + 'GMRbGP_' + write_name + '_stroke_' + str(
    #     stroke_index) + '_posteriors.pdf')

    # # plt.figure(figsize=(5, 4))
    # # plt.plot(Xt[:, 0], mu_gmr[:, 0], color=[0.20, 0.54, 0.93], linewidth=3.)
    # # miny = mu_gp_rshp[:, 0] - np.sqrt(sigma_gp_rshp[:, 0, 0])
    # # maxy = mu_gp_rshp[:, 0] + np.sqrt(sigma_gp_rshp[:, 0, 0])
    # # plt.fill_between(Xt[:, 0], miny, maxy, color=[0.83, 0.06, 0.06], alpha=0.3)
    # # for i in range(nb_posterior_samples):
    # # 	plt.plot(Xt[:, 0], mu_posterior[i][0], color=[0.64, 0., 0.65], linewidth=1.5)
    # # plt.plot(Xt[:, 0], mu_gp_rshp[:, 0], color=[0.83, 0.06, 0.06], linewidth=3)
    # # plt.scatter(X_obs[:, 0], Y_obs[:, 0], color=[0, 0, 0], zorder=60, s=100)
    # # axes = plt.gca()
    # # axes.set_ylim([-17., 17.])
    # # plt.xlabel('$t$', fontsize=30)
    # # plt.ylabel('$y_1$', fontsize=30)
    # # plt.tick_params(labelsize=20)
    # # plt.tight_layout()
    # # plt.savefig(file_name + '/figures/GMRbGP_B_posterior01_datasup.png')
    # #
    # # plt.figure(figsize=(5, 4))
    # # plt.plot(Xt[:, 0], mu_gmr[:, 1], color=[0.20, 0.54, 0.93], linewidth=3.)
    # # miny = mu_gp_rshp[:, 1] - np.sqrt(sigma_gp_rshp[:, 1, 1])
    # # maxy = mu_gp_rshp[:, 1] + np.sqrt(sigma_gp_rshp[:, 1, 1])
    # # plt.fill_between(Xt[:, 0], miny, maxy, color=[0.83, 0.06, 0.06], alpha=0.3)
    # # for i in range(nb_posterior_samples):
    # # 	plt.plot(Xt[:, 0], mu_posterior[i][1], color=[0.64, 0., 0.65], linewidth=1.5)
    # # plt.plot(Xt[:, 0], mu_gp_rshp[:, 1], color=[0.83, 0.06, 0.06], linewidth=3)
    # # plt.scatter(X_obs[:, 0], Y_obs[:, 1], color=[0, 0, 0], zorder=60, s=100)
    # # axes = plt.gca()
    # # axes.set_ylim([-17., 17.])
    # # plt.xlabel('$t$', fontsize=30)
    # # plt.ylabel('$y_2$', fontsize=30)
    # # plt.tick_params(labelsize=20)
    # # plt.tight_layout()
    # # plt.savefig(file_name + '/figures/GMRbGP_B_posterior02_datasup.png')
    
    # plt.show()