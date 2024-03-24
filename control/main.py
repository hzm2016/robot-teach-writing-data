import argparse
import os
# from motor_control import motor_control
from ctl_utils import *   
import time  

from scipy import interpolate

# path prediction
from utils.word_preprocess import *
 
sns.set(font_scale=1.5)
np.set_printoptions(precision=5)  


L_1 = 0.3
L_2 = 0.25
action_dim = 3
DIST_THREHOLD = 0.05  

FILE_FIG_NAME = './data/predicted_img/'
FILE_FONT_NAME = './data/font'
FILE_TRAIN_NAME = './data/training_path'
FILE_EVAL_NAME = './data/real_path'

ANGLE_1_RANGE = np.array([-1.90, 1.90])  
ANGLE_2_RANGE = np.array([-2.2, 2.5])
center_shift = np.array([0.15, -WIDTH / 2])

FONT_SIZE = 20

WRITING_Y = [-WIDTH / 2, WIDTH / 2]
WRITING_X = [0.13, 0.13 + WIDTH]

# initial angle (rad) :::
Initial_angle = np.array([-1.31, 1.527])  

Initial_point = np.array([0.32299, -0.23264])

Angle_initial = np.array([-0.314727, -0.122983, 1.981514]) 

# impedance params ::: 
Move_Impedance_Params = np.array([30.0, 30.0, 4.0, 0.2])  


def eval_stroke(
    stroke_points=None,  
    stroke_params=None,   
    target_point=Initial_point,  
    word_name='yi',   
    training_name='second',  
    stroke_index=0,   
    epi_time=0   
):
    Num_way_points = stroke_points.shape[0]
    
    initial_angle = np.zeros(2)
    initial_angle[0] = stroke_points[0, 0]
    initial_angle[1] = stroke_points[0, 1]
    start_point = forward_ik(initial_angle)
    
    # move to target point
    done = set_pen_up()
    done = move_to_target_point(start_point, Move_Impedance_Params, velocity=0.1)
    done = set_pen_down()
    
    # params_list = np.tile(impedance_params, (Num_way_points, 1))
    if stroke_params is None:
        exit()
    else:
        params_list = stroke_params
    
    folder_name = FILE_EVAL_NAME + '/' + training_name + '/' + word_name
    if os.path.exists(folder_name):
        pass
    else:
        os.makedirs(folder_name)
        
    stroke_angle_name = folder_name + '/' + 'real_angle_list_' + str(stroke_index) + '_' + str(
        epi_time) + '.txt'
    stroke_torque_name = folder_name + '/' + 'real_torque_list_' + str(stroke_index) + '_' + str(
        epi_time) + '.txt'
    done = motor_control.run_one_loop(
        stroke_points[:, 0].copy(), stroke_points[:, 1].copy(),
        params_list[:, 0].copy(), params_list[:, 1].copy(),
        params_list[:, 2].copy(), params_list[:, 3].copy(),
        Num_way_points,
        Angle_initial[0], Angle_initial[1],
        1,
        stroke_angle_name, stroke_torque_name
    )
    
    # move to target point
    done = set_pen_up()
    
    done = move_to_target_point(target_point, Move_Impedance_Params, velocity=0.1)
    
    print("Evaluate Stroke Once Done !!!")
    print("*" * 50)
    
    return done


def generate_training_path(
    word_name='mu',  
    eval_word_name='',   
    training_name='',   
    stroke_index=0,  
    epi_times=5,  
    training_times=5,  
    num_stroke=4,  
    plot=False   
):
    from forward_mode.utils.gmr import Gmr
    from forward_mode.utils.gmr import plot_gmm
    from forward_mode.utils.gp_coregionalize_with_mean_regression import GPCoregionalizedWithMeanRegression
    from forward_mode.utils.gmr_mean_mapping import GmrMeanMapping 
    from forward_mode.utils.gmr_kernels import Gmr_based_kernel  
    import GPy   

    folder_name = FILE_TRAIN_NAME + '/' + training_name + '/' + word_name  
    print('folder_name :', folder_name)   
    if os.path.exists(folder_name):   
        pass  
    else:   
        os.makedirs(folder_name)   
    
    dt = 0.01
    font_size = 30
    input_dim = 1  # time
    output_dim = 2  # x, y
    re_sample_index = int(60)   
    sample_factors = 0.001  

    # training hyper-parameters  
    in_idx = [0]  
    out_idx = [1, 2]   
    nb_states = 5  
    nb_prior_samples = 10  
    nb_posterior_samples = training_times 
    train_gmr = True   
    train_gp = True  

    # ====================== data processing ======================    
    # load from font
    train_word_path, _, _ = load_word_path(
        word_name=word_name,
        task_params=np.array([35, 35, 5, 0.5]),
        joint_params=np.array([35, 35, 5, 0.5]),
    )
    
    train_stroke_path = train_word_path[stroke_index]  
    stroke_list = Forward_list(
        stroke_path=train_stroke_path     
    )  
    Num_way_point = train_stroke_path.shape[0]   
    down_stroke_list, idx_list = fps(stroke_list, sample_factors)   
    idx_list = np.sort(idx_list)  
    down_stroke_list = down_stroke_list[idx_list]   
    down_stroke_list = np.array(down_stroke_list)    
    
    # load real path data
    word_path = load_real_word_path(
        root_path=FILE_EVAL_NAME,   
        word_name=eval_word_name,   
        training_name=training_name,  
        file_name='real_angle_list_',   
        epi_times=epi_times,  
        num_stroke=num_stroke,  
        plot=False
    )
    eval_stroke_path = np.array(word_path[stroke_index])
    # print("eval_stroke :", eval_stroke_path[:, 0, :])

    # =====================================================
    ################ deal with initial position ###########
    # =====================================================
    # offset_eval_stroke_path = np.zeros_like(eval_stroke_path)  
    for i in range(eval_stroke_path.shape[0]):
        # print('iteration :', i, eval_stroke_path[i, :, :].shape)  
        offset_value = eval_stroke_path[i, 0, :] - np.array([stroke_list[0, 0], stroke_list[0, 1], stroke_list[0, 0], stroke_list[0, 1]])
        eval_stroke_path[i, :, :] = eval_stroke_path[i, :, :] - offset_value
        # print(eval_stroke_path[i, 0, :] - np.array([stroke_list[0, 0], stroke_list[0, 1], stroke_list[0, 0], stroke_list[0, 1]]))
        # eval_stroke_path = eval_stroke_path[i, :, :] - (eval_stroke_path[i, 0, :] - np.array([stroke_list[0, 0], stroke_list[0, 1], stroke_list[0, 0], stroke_list[0, 1]]))

    # =====================================================  
    ############# process data before prediction ##########  
    # =====================================================  
    X_list, Y_list, X, Y, Xt, demos_np, nb_data = \
            eval_data_preprocess(
            # word_path,
            # stroke_index,
            stroke_path=eval_stroke_path,
            epi_times=epi_times,
            re_sample_index=re_sample_index,
            dt=dt,
            plot=False   
    )   
    T = Xt[-1][0]  
    print("Xt", T)  
    
    # ===================== generate new samples =============
    X_obs_list, Y_obs_list, X_obs, Y_obs = \
        load_new_obs_path(
            Num_way_point,  
            T,   
            down_stroke_list[:, 0].copy(),  
            down_stroke_list[:, 1].copy(),
            idx_list.copy()  
    )
    
    if train_gmr: 
        # ========================================================
        # ========================= GMM ==========================
        # ========================================================
        gmr_model = Gmr(nb_states=nb_states, nb_dim=input_dim + output_dim, in_idx=in_idx, out_idx=out_idx)
        gmr_model.init_params_kbins(demos_np.T, nb_samples=epi_times)
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
    
        if plot:
            plt.figure(figsize=(8, 8))
            for p in range(epi_times):
                plt.plot(Y[p * nb_data:(p + 1) * nb_data, 0], Y[p * nb_data:(p + 1) * nb_data, 1], color=[.7, .7, .7])
                plt.scatter(Y[p * nb_data, 0], Y[p * nb_data, 1], color=[.7, .7, .7], marker='X', s=50)
            plt.plot(mu_gmr[:, 0], mu_gmr[:, 1], color=[0.20, 0.54, 0.93], linewidth=3)
            plt.scatter(mu_gmr[0, 0], mu_gmr[0, 1], color=[0.20, 0.54, 0.93], marker='X', s=50)
            plot_gmm(mu_gmr, sigma_gmr, alpha=0.09, color=[0.20, 0.54, 0.93])
            plt.scatter(Y_obs[:, 0], Y_obs[:, 1], color=[0, 0, 0], zorder=60, s=100)
            # plt.scatter(X_obs_list, Y_obs_list, color=[0, 0, 0], zorder=60, s=100)
    
            axes = plt.gca()
            # axes.set_xlim([0.1, 0.45])
            # axes.set_ylim([-0.25, 0.25])
            # axes.set_xlim([-10, 10]) 
            axes.set_xlim([-20, 20]) 
            axes.set_ylim([-20, 20]) 
            # axes.set_ylim([-10, 10]) 
            # axes.set_xlim([-15, 15])
            # axes.set_ylim([-15, 15])
            plt.xlabel('$x(m)$', fontsize=font_size)
            plt.ylabel('$y(m)$', fontsize=font_size)
            plt.locator_params(nbins=3)
            plt.tick_params(labelsize=font_size)
            plt.tight_layout()
            plt.title(word_name, fontsize=font_size)
            plt.savefig(folder_name + '/' + 'GMR_' + eval_word_name + '_stroke_' + str(stroke_index) + '.pdf')
    
            plt.show()

    if train_gp: 
        # ========================================================
        # ========================= GPR ==========================
        # ========================================================
        # Define GPR likelihood and kernels ::: original : 0.01
        nb_data_test = Xt.shape[0]
        Xtest, _, output_index = GPy.util.multioutput.build_XY([np.hstack((Xt, Xt)) for i in range(output_dim)])
    
        likelihoods_list = [GPy.likelihoods.Gaussian(name="Gaussian_noise_%s" %j, variance=1) for j in range(output_dim)]
        kernel_list = [GPy.kern.Matern52(1, variance=0.1, lengthscale=5) for i in range(gmr_model.nb_states)]
        # kernel_list = [GPy.kern.RBF(1, variance=1, lengthscale=0.5) for i in range(gmr_model.nb_states)]
        # kernel_list = [GPy.kern.RBF(1, variance=1, lengthscale=5) for i in range(gmr_model.nb_states)]
    
        # Fix variance of kernels
        for kernel in kernel_list:
            kernel.variance.fix(1.0)
            kernel.lengthscale.constrain_bounded(0.01, 10.)
        
        # Bound noise parameters
        for likelihood in likelihoods_list:
            likelihood.variance.constrain_bounded(0.01, 0.5)
    
        # GPR model
        K = Gmr_based_kernel(gmr_model=gmr_model, kernel_list=kernel_list)
        mf = GmrMeanMapping(2 * input_dim + 1, 1, gmr_model)
        m = GPCoregionalizedWithMeanRegression(
            X_list, Y_list,
            kernel=K,
            likelihoods_list=likelihoods_list,
            mean_function=mf
        )
    
        # Parameters optimization
        m.optimize('bfgs', max_iters=200, messages=True)
    
        # Print model parameters
        print(m)
    
        # GPR prior (no observations)
        prior_traj = []  
        prior_mean = mf.f(Xtest)[:, 0]  
        prior_kernel = m.kern.K(Xtest)  
        for i in range(nb_prior_samples):  
            # print("prior_kernel :", prior_kernel.shape)
            prior_traj_tmp = np.random.multivariate_normal(prior_mean, prior_kernel)
            # print("prior_traj_tmp :", prior_traj_tmp.shape)
            prior_traj.append(np.reshape(prior_traj_tmp, (output_dim, -1)))
        prior_kernel_tmp = np.zeros((nb_data_test, nb_data_test, output_dim * output_dim))
        for i in range(output_dim):
            for j in range(output_dim):
                prior_kernel_tmp[:, :, i * output_dim + j] = prior_kernel[i * nb_data_test:(i + 1) * nb_data_test, j * nb_data_test:(j + 1) * nb_data_test]
        prior_kernel_rshp = np.zeros((nb_data_test, output_dim, output_dim))
        for i in range(nb_data_test):
            prior_kernel_rshp[i] = np.reshape(prior_kernel_tmp[i, i, :], (output_dim, output_dim))
    
        # GPR posterior -> new points observed (the training points are discarded as they are "included" in the GMM)
        m_obs = \
            GPCoregionalizedWithMeanRegression(
                X_obs_list, Y_obs_list, kernel=K,
                likelihoods_list=likelihoods_list,
                mean_function=mf
            )
        mu_posterior_tmp = \
            m_obs.posterior_samples_f(
                Xtest, full_cov=True, size=nb_posterior_samples
            )
    
        mu_posterior = []
        for i in range(nb_posterior_samples):
            mu_posterior.append(np.reshape(mu_posterior_tmp[:, 0, i], (output_dim, -1)))
    
        # GPR prediction
        mu_gp, sigma_gp = m_obs.predict(Xtest, full_cov=True, Y_metadata={'output_index': output_index})
        mu_gp_rshp = np.reshape(mu_gp, (output_dim, -1)).T
        sigma_gp_tmp = np.zeros((nb_data_test, nb_data_test, output_dim * output_dim))
        for i in range(output_dim):
            for j in range(output_dim):
                sigma_gp_tmp[:, :, i * output_dim + j] = sigma_gp[i * nb_data_test:(i + 1) * nb_data_test, j * nb_data_test:(j + 1) * nb_data_test]
        sigma_gp_rshp = np.zeros((nb_data_test, output_dim, output_dim))
        for i in range(nb_data_test):
            sigma_gp_rshp[i] = np.reshape(sigma_gp_tmp[i, i, :], (output_dim, output_dim))
        # print("sigma_gp_rshp", sigma_gp_rshp.shape)
    
        if plot:
            # Posterior
            plt.figure(figsize=(8, 8))
            plt.plot(mu_gmr[:, 0], mu_gmr[:, 1], color=[0.20, 0.54, 0.93], linewidth=3.)
            plot_gmm(mu_gmr, sigma_gmr, alpha=0.09, color=[0.20, 0.54, 0.93])
            # plot_gmm(mu_gp_rshp, sigma_gp_rshp, alpha=0.05, color=[0.83, 0.06, 0.06])
            for i in range(nb_posterior_samples):
                plt.plot(mu_posterior[i][0], mu_posterior[i][1], color=[0.64, 0., 0.65], linewidth=1.5)
                plt.scatter(mu_posterior[i][0, 0], mu_posterior[i][1, 0], color=[0.64, 0., 0.65], marker='X', s=80)
            # plt.plot(mu_gp_rshp[:, 0], mu_gp_rshp[:, 1], color=[0.83, 0.06, 0.06], linewidth=3.)
            # plt.scatter(mu_gp_rshp[0, 0], mu_gp_rshp[0, 1], color=[0.83, 0.06, 0.06], marker='X', s=80)
    
            plt.scatter(Y_obs[:, 0], Y_obs[:, 1], color=[0, 0, 0], zorder=60, s=100)
    
            axes = plt.gca()
            axes.set_xlim([-20, 20])
            # axes.set_xlim([-10, 10])
            # axes.set_xlim([-10, 10])
            axes.set_ylim([-20, 20])
            # axes.set_ylim([-10, 10])
            # axes.set_xlim([-15, 15])
            # axes.set_ylim([-15, 15])
            plt.xlabel('$x(m)$', fontsize=font_size)
            plt.ylabel('$y(m)$', fontsize=font_size)
            plt.locator_params(nbins=3)
            plt.tick_params(labelsize=font_size)
            plt.tight_layout()
            plt.savefig(folder_name + '/' + 'GMRbGP_' + eval_word_name + '_stroke_' + str(
                stroke_index) + '_posteriors.pdf')
    
            plt.show()
        
        print("generated trajectories :", np.array(mu_posterior).shape)
        np.save(folder_name + '/training_stroke_' + eval_word_name + '_' + str(stroke_index) + '_samples.npy', np.array(mu_posterior))
        return mu_posterior


def training_samples_to_waypoints(
    word_name='mu',  
    eval_word_name='',  
    training_name='',  
    stroke_index=0,  
    Num_waypoints=10000,  
    sample_index=0,  
    task_params=None,   
    joint_params=None,  
    desire_angle_list=None,   
    plot=True  
):
    print("============== {} ============".format('Load Training Samples !!!'))
    folder_name = FILE_TRAIN_NAME + '/' + word_name + '/' + training_name + '/training_stroke_' + eval_word_name + '_' + str(stroke_index) + '_samples.npy'
    training_samples = np.load(folder_name)  
    angle_list = np.zeros((Num_waypoints, 2))  
    
    nb_posterior_samples = training_samples.shape[0]
    data_sample = training_samples.shape[2]
    index = np.linspace(0, data_sample-1, data_sample)
    index_list = np.linspace(0, data_sample-1, Num_waypoints)
    print("training_samples :", training_samples.shape, data_sample, index.shape, training_samples[sample_index][0].shape)
    
    # x_list = np.linspace(training_samples[0][0][0], training_samples[0][0][-1], Num_waypoints)
    # x_list = np.linspace(training_samples[0][0][0], training_samples[0][0][-1], Num_waypoints)
    x_list_ori, y_list_ori = de_scale_translate_process_main(training_samples[sample_index][0], training_samples[sample_index][1])
    
    fx = interpolate.interp1d(index, x_list_ori, kind='linear')
    fy = interpolate.interp1d(index, y_list_ori, kind='linear')
    x_list = fx(index_list)
    y_list = fy(index_list)
    
    trajectory_list = Forward_list(stroke_path=desire_angle_list)
    print(trajectory_list)
    # path_data[:, 0] = savgol_filter(traj[:, 0], filter_size, 3, mode='nearest')
    # path_data[:, 1] = savgol_filter(traj[:, 1], filter_size, 3, mode='nearest')
    
    if plot:
        # Posterior
        plt.figure(figsize=(8, 8))
        # for i in range(nb_posterior_samples):
        #     plt.plot(training_samples[i][0], training_samples[i][1], color=[0.64, 0., 0.65], linewidth=1.5)
        #     plt.scatter(training_samples[i][0, 0], training_samples[i][1, 0], color=[0.64, 0., 0.65], marker='X', s=80)
        plt.plot(x_list, y_list)
        plt.plot(trajectory_list[:, 0], trajectory_list[:, 1], color='black', linewidth=3)
        axes = plt.gca()
        axes.set_xlim(WRITING_X)
        axes.set_ylim(WRITING_Y)
        # # axes.set_xlim([-10, 10])
        # axes.set_ylim([-10, 10])
        # axes.set_xlim([-15, 15])
        # axes.set_ylim([-15, 15])
        plt.xlabel('$x(m)$', fontsize=FONT_SIZE)
        plt.ylabel('$y(m)$', fontsize=FONT_SIZE)
        plt.locator_params(nbins=3)
        plt.tick_params(labelsize=FONT_SIZE)
        plt.tight_layout()
        plt.title(word_name, fontsize=FONT_SIZE)
        # plt.savefig(FILE_FIG_NAME + '/' + word_name + '/' + 'GMRbGP_' + word_name + '_stroke_' + str(
        #     stroke_index) + '_posteriors.pdf')
        plt.show()
    
    for i in range(Num_waypoints):
        angle = IK(np.array([x_list[i], y_list[i]]))
        angle_list[i, :] = angle
    
    max_angle_1 = np.max(angle_list[:, 0])
    max_angle_2 = np.max(angle_list[:, 1])
    print("Max angle 1 (rad) :", max_angle_1)
    print("Max angle 2 (rad):", max_angle_2)
    
    if max_angle_1 < ANGLE_1_RANGE[0] or max_angle_1 > ANGLE_1_RANGE[1]:
        print("!!!!!! angle 1 is out of range !!!!!")
        print("max angle 1 :::", max_angle_1)
        exit()

    if max_angle_2 < ANGLE_2_RANGE[0] or max_angle_2 > ANGLE_2_RANGE[1]:
        print("!!!!!! angle 1 is out of range !!!!!")
        print("max angle 2 :::", max_angle_2)
        exit()

    print("angle_list_shape :", angle_list.shape)
    
    joint_params_list = load_impedance_list(
        word_name=word_name,
        stroke_index=stroke_index,
        desired_angle_list=angle_list,
        current_angle_list=angle_list,
        joint_params=joint_params,
        task_params=task_params
    )
    return angle_list, joint_params_list



def main(args): 

    if args.test:

        theta_1 = motor_control.read_initial_angle_1()
        print("theta_1 :", theta_1)
        theta_2 = motor_control.read_initial_angle_2()
        print("theta_2 :", theta_2)
        theta_3 = motor_control.read_initial_angle_3()
        print("theta_3 :", theta_3)

        angle, point = get_observation()
        # print("angle :", angle)
        print("point :", point)

        set_pen_up()

        set_pen_down()

        motor_stop()

        # target_point = np.array([0.35, -0.15])
        # move_to_target_point(
        #     target_point,
        #     # impedance_params=np.array([420.0, 420.0, 15.5, 15.5]),
        #     impedance_params=np.array([40.0, 40.0, 5.0, 0.2]),
        #     velocity=0.1  
        # )  
        
        # time.sleep(0.2)  
        # angle, point = get_observation()  
        # # print("angle :", angle)
        # print("point :", point)

        # stiff_joint, damping_joint = Stiff_convert(np.array([0.5, 0.5])
        # np.diag([40, 40]), np.diag([1.0, 1.0]))
        # print("stiff_joint :", stiff_joint)
        # print("damping_joint :", damping_joint)
        # motor_control.Convert_stiffness(40.0, 40.0, 0.5, 0.5)
    
    if args.assist:
        eval_times = 1
        word_path, word_joint_params, word_task_params = load_word_path(
            word_name=args.word_name,
            task_params=np.array([35, 35, 5, 0.5]),
            joint_params=np.array([40, 40, 5, 0.5]),
            )
        
        word_params = word_joint_params  
        # evaluation writing
        for i in range(eval_times):   
            # word
            write_word(word_path, word_params=word_params, word_name=args.word_name, epi_times=i)
            # stroke

    if args.sample:
        # generate new training samples
        stroke_training_samples = generate_training_path(
            word_name=args.word_name,   
            eval_word_name=args.eval_word_name,   
            training_name=args.training_name,   
            training_times=args.training_times,  
            stroke_index=args.stroke_index,  
            epi_times=args.eval_times, 
            num_stroke=1,
            plot=True
        )
    
    if args.eval:

        stiffness = 5.0
        damping = 1.0
        joint_params = np.array([stiffness, stiffness, damping, 0.5])  
        task_params = np.array([stiffness, stiffness, damping, 0.5])  
        
        # eval_times = 1
        word_path, word_joint_params, word_task_params = load_word_path(
            word_name=args.word_name,  
            task_params=task_params,  
            joint_params=joint_params,   
        )

        angle_list = word_path[args.stroke_index]   
        Num_waypoints = angle_list.shape[0]   
        print("word_one_stroke_num_way_points :", Num_waypoints)  
        
        # joint_params_list = load_impedance_list(
        #     word_name=args.word_name,
        #     stroke_index=args.stroke_index,
        #     desired_angle_list=angle_list,
        #     current_angle_list=angle_list,
        #     task_params=np.array([35, 35, 5, 0.5]),
        #     joint_params=np.array([35, 35, 5, 0.5])
        # )
        
        # evaluation writing
        for i in range(args.eval_times):
            # write_word(
            #     word_path, 
            #     word_params=word_params, 
            #     word_name=write_name, 
            #     epi_times=i
            # )
            
            # eval_stroke(
            #     stroke_points=word_path[args.stroke_index],
            #     stroke_params=word_joint_params[args.stroke_index],
            #     target_point=Initial_point,
            #     word_name=args.word_name + '_5', 
            #     stroke_index=args.stroke_index,  
            #     epi_time=i  
            # )

            stroke_points, joint_params_list = \
            training_samples_to_waypoints(   
                word_name=args.word_name,  
                eval_word_name=args.eval_word_name,  
                training_name=args.training_name,  
                stroke_index=args.stroke_index,  
                Num_waypoints=Num_waypoints,  
                sample_index=i,  
                task_params=task_params,  
                joint_params=joint_params,   
                desire_angle_list=angle_list,   
                plot=False
            )   

            eval_stroke(
                stroke_points=stroke_points,   
                stroke_params=joint_params_list,   
                target_point=Initial_point,   
                word_name=args.save_word_name,   
                training_name=args.training_name,   
                stroke_index=args.stroke_index,   
                epi_time=i   
            )  

        motor_stop()  

    if args.plot:  

        # plot_real_stroke_2d_path(
        #     root_path='./data/font_data/xing/',
        #     file_name='angle_list_5',
        #     stroke_num=5, 
        #     delimiter=' ', 
        #     skiprows=1 
        # )  

        # plot_real_2d_path(
        #     root_path=FILE_EVAL_NAME + '/' + args.training_name + '/' + args.word_name + '/',
        #     file_name='real_angle_list_', 
        #     stroke_num=1, 
        #     epi_times=args.eval_times,  
        #     delimiter=',',  
        #     skiprows=1  
        # )
        
        # word_path, word_joint_params, word_task_params= \
        #     load_word_path(
        #         word_name=write_name,
        #         task_params=np.array([150, 150, 1, 0.5]),
        #         joint_params=np.array([20, 20, 1, 0.5])
        # )
        
        # print("task", np.array(word_joint_params[0]).shape)
        # plot_impedance_path(
        #     word_joint_params=word_joint_params,
        #     word_task_params=word_task_params,
        #     line_width=3.0
        # )

        # plot_torque_path(
        #     root_path=FILE_EVAL_NAME + '/' + args.word_name + '/',
        #     file_name='real_torque_list_',
        #     stroke_num=1,
        #     epi_time=args.eval_times,
        #     delimiter=',',
        #     skiprows=1,
        #     render=True
        # )

        # plot_velocity_path(
        #     root_path=FILE_EVAL_NAME + '/' + args.word_name + '/',
        #     file_name='real_angle_list_',
        #     stroke_num=1,
        #     epi_time=0,
        #     delimiter=',',
        #     skiprows=1
        # )

        # plot_real_error_path(
        #     root_path='./data/font_data/' + write_name + '/',
        #     file_name='real_angle_list_',
        #     stroke_num=6,
        #     epi_num=5,
        #     delimiter=' ',
        #     skiprows=1
        # )

        plot_real_error_path_comparison(
            root_path=FILE_EVAL_NAME + '/second',
            folder_name=args.word_name,
            file_name='real_angle_list_',
            stroke_num=1,
            epi_num=5,
            delimiter=',',
            skiprows=1
        )

        # plot_torque_comparison(
        #     root_path=FILE_EVAL_NAME + '/second_time',
        #     folder_name=args.word_name,
        #     file_name='real_torque_list_',
        #     stroke_num=1,
        #     epi_time=5,
        #     delimiter=',',
        #     skiprows=1
        # )

        # plot_real_2d_demo_path(
        # root_path='',
        # file_name=write_name,
        # delimiter=',',
        # skiprows=1
        # )

    if args.generate_path:    
        traj = np.loadtxt(FILE_FONT_NAME + '/' + args.word_name + '/句_1_font1.txt')   
        print("traj :", traj)   
        generate_stroke_path(
            traj, inter_type=1, inverse=True,  
            center_shift=np.array([-WIDTH/2, 0.15]),  
            velocity=0.04, Ts=0.001, filter_size=13,  
            plot_show=True, save_path=True, word_name='ju', stroke_name=1  
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--test', action='store_true', default=False, help='hardware design')    
    parser.add_argument('--assist', type=bool, default=False, help='assist mode')    
    parser.add_argument('--eval', action='store_true', default=False, help='evaluate writing results')   
    parser.add_argument('--plot', action='store_true', default=False, help='whether plot results')   
    parser.add_argument('--sample', action='store_true', default=False, help='whether sample new training results')   
    parser.add_argument('--generate_path', type=bool, default=False, help='whether plot results')   

    parser.add_argument('--word_name', type=str, default='yi', help='give write word name')  
    parser.add_argument('--eval_word_name', type=str, default='yi', help='give write word name')  
    parser.add_argument('--save_word_name', type=str, default='yi_5_5', help='give write word name')    
    parser.add_argument('--stroke_index', type=int, default=0, help='give write word name')    
    parser.add_argument('--sample_index', type=int, default=0, help='give write word name')   

    parser.add_argument('--file_name', type=str, default='real_angle_list_', help='give write word name')   
    parser.add_argument('--training_name', type=str, default='second', help='give write word name')   
    parser.add_argument('--eval_times', type=int, default=5, help='give write word name')   
    parser.add_argument('--training_times', type=int, default=5, help='give write word name')   

    args = parser.parse_args()   
    
    main(args)  