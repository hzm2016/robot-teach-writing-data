import numpy as np
np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt
import math
import seaborn as sns
sns.set_theme()
sns.set(font_scale=0.5)
from scipy import interpolate
from .plot_path import * 

import cv2

from scipy.signal import savgol_filter
from .utils import IK, Stiff_convert, rotate_point
from .utils import *

# writing space
WIDTH = 0.360
HEIGHT = 0.360

# image size
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128

# joints limits
action_dim = 2
ANGLE_1_RANGE = np.array([-1.90, 1.90])
ANGLE_2_RANGE = np.array([-2.2, 2.5])
center_shift = np.array([0.15, -WIDTH / 2])

Length = [0.30, 0.150, 0.25, 0.125]
L1 = Length[0]
L2 = Length[2]

Ts = 0.001

# plot parameters
linewidth = 3


def forward_ik_path(
        angle_list,
        transfer_to_img=True
):
    """
        calculate osc point
    """
    point_list = np.zeros_like(angle_list)

    for i in range(point_list.shape[0]):
        point_list[i, 0] = L1 * math.cos(angle_list[i, 0]) + L2 * math.cos(angle_list[i, 0] + angle_list[i, 1])
        point_list[i, 1] = L1 * math.sin(angle_list[i, 0]) + L2 * math.sin(angle_list[i, 0] + angle_list[i, 1])

    point_list = point_list - center_shift
    if transfer_to_img:
        point_list[:, 0] *= 128/WIDTH
        point_list[:, 1] *= 128/HEIGHT

    return point_list


def transform_task_space(x_list, y_list, offset=np.array([0.0, 0.0])):
    new_osc_data = []
    for i in range(len(x_list)):
        ori_data = np.ones((x_list[i].shape[0], 3))
        ori_data[:, 0] = x_list[i]
        ori_data[:, 1] = y_list[i]
        T_matrix = np.array([[1, 0, 0], [0, 1, 0], [offset[0], offset[1], 1]])
        new_data = ori_data.dot(T_matrix)
        new_osc_data.append(new_data)
    return new_osc_data


def generate_stroke_stiffness_path(
    angle_list, stiffness_list, damping_list,
    save_path=False,
    save_root='',
    word_name='yi', 
    stroke_index=0
):  
    """
        :param angle_list:
        :param stiffness_list:
        :param damping_list:
        :return:
    """
    stiff_joint_list = []
    damping_joint_list = []
    for i in range(angle_list.shape[0]):
        stiff_task = np.diag(stiffness_list[i, :]).copy()
        damping_task = np.diag(damping_list[i, :]).copy()
        
        stiff_joint, damping_joint = Stiff_convert(angle_list[i, :], stiff_task, damping_task)  
        
        # calculate stiffness and damping in joint space
        stiff_joint_list.append([stiff_joint[0, 0], stiff_joint[1, 1]])
        damping_joint_list.append([damping_joint[0, 0], damping_joint[1, 1]])

    params_list = np.hstack((stiff_joint_list, damping_joint_list))   
    print("params_list :", params_list.shape)   
    if save_path:
        np.savetxt(save_root + '/' + word_name + '/' + 'params_stroke_list_' + str(stroke_index) + '.txt',
                   params_list, fmt='%.05f')

    return params_list


def generate_stroke_path_new(
    traj, inter_type=1, inverse=True,  
    center_shift=np.array([-WIDTH/2, 0.23]),  
    velocity=0.04, Ts=0.001, filter_size=17,  
    plot_show=False, save_path=False, word_name=None, stroke_name=0
):
    data_sample = traj.shape[0]
    print(data_sample) 
    x_list_ori = traj[:, 1]
    y_list_ori = traj[:, 0] 
    Num_waypoints = 1000
    index = np.linspace(0, data_sample-1, data_sample)
    index_list = np.linspace(0, data_sample-1, Num_waypoints)

    fx = interpolate.interp1d(index, x_list_ori, kind='linear')
    fy = interpolate.interp1d(index, y_list_ori, kind='linear')
    x_list = fx(index_list)
    y_list = fy(index_list)

    plot_ori_osc_2d_path(
    x_list,
    y_list, 
    linewidth=1,  
)


def generate_stroke_path(
    traj, inter_type=1, inverse=True,  
    center_shift=np.array([-WIDTH/2, 0.23]),  
    velocity=0.02, Ts=0.001, filter_size=17,  
    plot_show=False, save_path=False, word_name=None, stroke_name=0
):
    """
         generate stroke trajectory from list
         velocity ::: 0.04m/s
    """
    # calculate length of path
    dist = 0.0
    for i in range(len(traj) - 1): 
        point_1 = np.array([traj[i, 1], traj[i, 0]])  
        point_2 = np.array([traj[i+1, 1], traj[i+1, 0]])
        dist += np.linalg.norm((point_2.copy() - point_1.copy()), ord=2)

    path_data = np.zeros_like(traj)

    path_data[:, 0] = savgol_filter(traj[:, 0], filter_size, 3, mode='nearest')
    path_data[:, 1] = savgol_filter(traj[:, 1], filter_size, 3, mode='nearest')

    # M = N//len(traj)
    # x_list = []
    # y_list = []
    # for i in range(len(traj)):
    #     # need to check which dimension can be applied for interp
    #     x_list_i = np.linspace(path_data[i, 1], path_data[i+1, 1], M)
    #     y_list_i = np.interp(x_list_i, path_data[i:i+1, 1], path_data[i:i+1, 0])
    #     x_list.append(x_list_i)
    #     y_list.append(y_list_i)

    # # need to check which dimension can be applied for interp
    # x_list = np.linspace(path_data[-1, 1], path_data[0, 1], N)
    # # x_list = path(1, 2):(path(end, 2) - path(1, 2)) / (N - 1): path(end, 2)
    # y_list = np.interp(x_list, path_data[:, 1][::-1], path_data[:, 0][::-1])

    # transform to work space
    ratio = IMAGE_WIDTH / WIDTH

    period = dist/ratio/velocity
    print("Distance (mm) :", np.array(dist))
    print("Period (s) :", np.array(period))
    N = np.array(period / Ts).astype(int)

    # start_point = np.array([path_data[0, 1], path_data[0, 0]])
    # end_point = np.array([path_data[-1, 1], path_data[-1, 0]])
    # dir = end_point - start_point
    # angle = math.atan2(dir[1], dir[0])

    # if angle > -math.pi/4 and angle < 0:
    #     inter_type = 2
    #     inverse = False
    # if angle > 3.0 or angle < -3.0:
    #     inter_type = 2
    # if angle > -math.pi/2 and angle < - math.pi/4:
    #     inter_type = 1
    # if angle > math.pi/4 and angle < math.pi *3/4:
    #     inverse = False
    # if angle > math.pi *3/4 and angle < math.pi:
    #     inter_type = 2

    # sample_x = []
    # sample_y = []
    # if inter_type==1:
    #     if inverse:
    #         y_list = np.linspace(path_data[-1, 0], path_data[0, 0], N)
    #         x_list = np.interp(y_list, path_data[:, 0][::-1], path_data[:, 1][::-1])
    #     else:
    #         y_list = np.linspace(path_data[0, 0], path_data[-1, 0], N)
    #         x_list = np.interp(y_list, path_data[:, 0], path_data[:, 1])
    #     # sample_y = np.array(path_data[:, 0])
    #     # sample_x = np.array(path_data[:, 1])
    #     #
    #     # # 进行三次样条拟合
    #     # ipo3 = spi.splrep(sample_y, sample_x, k=3)  # 样本点导入，生成参数
    #     # x_list = spi.splev(y_list, ipo3)  # 根据观测点和样条参数，生成插值
    # elif inter_type==2:
    #     if inverse:
    #         x_list = np.linspace(path_data[-1, 1], path_data[0, 1], N)
    #         y_list = np.interp(x_list, path_data[:, 1][::-1], path_data[:, 0][::-1])
    #     else:
    #         x_list = np.linspace(path_data[0, 1], path_data[-1, 1], N)
    #         y_list = np.interp(x_list, path_data[:, 1], path_data[:, 0])
    # else:
    #     print("Please check the given stroke path !!!") 

    data_sample = traj.shape[0]
    x_list_ori = traj[:, 1]
    y_list_ori = traj[:, 0] 
    index = np.linspace(0, data_sample-1, data_sample)
    index_list = np.linspace(0, data_sample-1, N)

    fx = interpolate.interp1d(index, x_list_ori, kind='linear')
    fy = interpolate.interp1d(index, y_list_ori, kind='linear')
    x_list = fx(index_list)
    y_list = fy(index_list)

    image_points = np.vstack((x_list, y_list)).transpose()

    x_1_list = x_list/ratio + center_shift[1]
    x_2_list = y_list/ratio + center_shift[0]

    x_1_list = x_1_list[::-1]
    x_2_list = x_2_list[::-1]

    task_points = np.vstack((x_1_list, x_2_list)).transpose() 

    angle_1_list_e = []
    angle_2_list_e = []

    for t in range(1, N):
        x1 = x_1_list[t]
        x2 = x_2_list[t]

        point = np.array([x1, x2])

        angle = IK(point)

        # Inverse kinematics
        angle_1_list_e.append(np.round(angle[0].copy(), 5))
        angle_2_list_e.append(np.round(angle[1].copy(), 5))

    max_angle_1 = np.max(angle_1_list_e)
    max_angle_2 = np.max(angle_2_list_e)
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

    way_points = np.vstack((angle_1_list_e, angle_2_list_e)).transpose()
    print('+' * 50)
    print("Check success with way_points :", way_points.shape[0])

    if plot_show:
        plot_stroke_path(period, traj, image_points, task_points, way_points)

    if save_path:
        np.savetxt('data/font_data/' + word_name + '/' + 'angle_list_' + str(stroke_name) + '.txt', way_points, fmt='%.05f')

    return way_points, image_points, task_points, period


def generate_word_path(
        traj_list,
        stiffness,
        damping,
        inter_list=None,  
        inverse_list=None,  
        center_shift=np.array([0.23, -WIDTH/2]),  
        velocity=0.04,  
        filter_size=17,  
        plot_show=False,  
        save_path=False,  
        save_root='control/font_data',  
        word_name='tian'):  
    """
        generate word path
    """
    word_angle_list = []
    word_image_points = []
    word_task_points = []
    period_list = []
    word_params_list = []
    for stroke_index in range(len(traj_list)):
        """ get one stroke """
        traj = traj_list[stroke_index]
        print('=' * 20)
        print('stroke index :', stroke_index)
        stroke_angle_list, stroke_image_points, stroke_task_points, period \
            = generate_stroke_path(
                traj,
                inter_type=inter_list[stroke_index],
                inverse=inverse_list[stroke_index],
                center_shift=center_shift,
                velocity=velocity,
                filter_size=filter_size,
                plot_show=False,
                save_path=save_path,
                word_name=word_name,
                stroke_name=stroke_index
            )

        stiffness_list = np.tile(stiffness, (stroke_angle_list.shape[0], 1))
        damping_list = np.tile(damping, (stroke_angle_list.shape[0], 1))
        params_list = generate_stroke_stiffness_path(stroke_angle_list, stiffness_list, damping_list,
                                       save_path=save_path, save_root=save_root, word_name=word_name, stroke_name=stroke_index)

        word_angle_list.append(stroke_angle_list)
        word_image_points.append(stroke_image_points)
        word_task_points.append(stroke_task_points.copy())
        period_list.append(period)
        word_params_list.append(params_list[:, :2])

    if plot_show:
        plot_word_path(period_list, traj_list, word_image_points, word_task_points, word_angle_list,
                       word_folder=save_root,
                       word_name=word_name
                       )  
        # plot_torque(word_params_list, period_list)