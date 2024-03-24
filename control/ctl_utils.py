import argparse
import os
from motor_control import motor_control
import time
import numpy as np

# path prediction
from utils.word_preprocess import * 

L_1 = 0.3 
L_2 = 0.25
action_dim = 3
DIST_THREHOLD = 0.05

FILE_FIG_NAME = './data/predicted_images/'
FILE_FONT_NAME = './data/font_data'
FILE_TRAIN_NAME = './data/training_data'
FILE_EVAL_NAME = './data/real_path_data'

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


def set_pen_up():
    """
        pull pen up
    """
    # motor_control.motor_3_stop()
    up_angle = np.int32(9000)
    done = motor_control.set_position(0.0, up_angle)
    time.sleep(1.0)

    return done


def set_pen_down():
    """
        pull pen down
    """
    # motor_control.motor_3_stop()
    down_angle = np.int32(11200)
    done = motor_control.set_position(0.0, down_angle)
    time.sleep(2.0)

    return done


def motor_stop():
    """
        sometimes need to stop motors
    """
    motor_control.motor_two_link_stop()
    motor_control.motor_3_stop()


def get_observation(angle_initial=Angle_initial):
    """
        obtain joint angles and cartesian state
    """
    # ######################################################
    # ############## get current state #####################
    # ######################################################
    angle = np.zeros(action_dim)
    point = np.zeros(action_dim)
    
    print('+' * 20)
    angle[0] = motor_control.read_angle_1(angle_initial[0])
    # print("Joint 1 angles (rad) :", angle[0])
    angle[1] = motor_control.read_angle_2(angle_initial[1], angle[0].copy())
    # print("Joint 2 angles (rad) :", angle[1])
    # angle[2] = motor_control.read_angle_3(angle_initial[2])
    print("Joint angles (rad) :", np.array(angle))
    
    point[0] = L_1 * math.cos(angle[0]) + L_2 * math.cos(angle[0] + angle[1])
    point[1] = L_1 * math.sin(angle[0]) + L_2 * math.sin(angle[0] + angle[1])
    print("Position (m) :", np.array(point))
    
    return angle, point


def reset_and_calibration():
    print("Please make sure two links are at zero position !!!")
    angle_initial = np.zeros(3)
    
    angle_initial[0] = motor_control.read_initial_angle_1()
    angle_initial[1] = motor_control.read_initial_angle_2()
    angle_initial[2] = motor_control.read_initial_angle_3()
    
    return angle_initial


def load_word_path(
    root_path='./data/font_data',
    word_name=None,
    task_params=None,
    joint_params=None
):
    # load original training way points :::
    word_file = root_path + '/' + word_name + '/'
    stroke_list_file = glob.glob(word_file + 'angle_list_*txt')
    print("Load stroke data %d", len(stroke_list_file))

    word_path = []
    word_joint_params = []
    word_task_params = []
    
    for i in range(len(stroke_list_file)):
        way_points = np.loadtxt(word_file + 'angle_list_' + str(i) + '.txt', delimiter=' ')

        if joint_params is not None:
            joint_params_list = np.tile(joint_params, (way_points.shape[0], 1))
        else:
            joint_params_list = np.loadtxt(word_file + 'params_list_' + str(i) + '.txt', delimiter=' ')
         
        N_way_points = way_points.shape[0]
        print("N_way_points :", N_way_points)

        word_path.append(way_points.copy())
        word_joint_params.append(joint_params_list.copy())
    
        # word parameters
        if task_params is not None:
            task_params_list = np.tile(task_params, (way_points.shape[0], 1))
        else:
            task_params_list = np.loadtxt(word_file + 'params_list_' + str(i) + '.txt', delimiter=' ')

        angle_list = way_points
        stiffness_list = task_params_list[:, :2]
        damping_list = task_params_list[:, 2:]
        joint_params_list = generate_stroke_stiffness_path(
            angle_list, stiffness_list, damping_list,
            save_path=False, save_root='', word_name='yi', stroke_index=0
        )
        
        word_task_params.append(joint_params_list)

    return word_path, word_joint_params, word_task_params



def load_impedance_list(
    word_name='mu', 
    stroke_index=0, 
    desired_angle_list=None, 
    current_angle_list=None, 
    joint_params=None,  
    task_params=None  
):
    print("============== {} ============".format('Load Impedance !!!'))
    way_points = desired_angle_list  
    N_way_points = way_points.shape[0]  
    
    # joint parameters
    if joint_params is not None:
        joint_params_list = np.tile(joint_params, (way_points.shape[0], 1))
    else:  
        joint_params_list = np.loadtxt(FILE_TRAIN_NAME + '/' + word_name + '/' + 'params_stroke_list_' + str(stroke_index) + '.txt', delimiter=' ')
    
    # task parameters
    if task_params is not None:
        task_params_list = np.tile(task_params, (way_points.shape[0], 1))
    else:
        task_params_list = np.loadtxt(FILE_TRAIN_NAME + '/' + word_name + '/' + 'params_stroke_list_' + str(stroke_index) + '.txt', delimiter=' ')
    
    # stiffness_list = task_params_list[:, :2] 
    # damping_list = task_params_list[:, 2:] 
    # joint_params_list = generate_stroke_stiffness_path(
    #     desired_angle_list, stiffness_list, damping_list,
    #     save_path=False, save_root=FILE_TRAIN_NAME, word_name=word_name, stroke_index=stroke_index
    # )
    
    return joint_params_list


def get_demo_writting():
    """
        zero impedance control
    """
    buff_size = np.zeros((100, 2))
    impedance_params = np.array([35.0, 25.0, 0.4, 0.1])

    set_pen_down()
    motor_control.get_demonstration(Angle_initial[0], Angle_initial[1],
    2.0, 2.0, 0.0, 0.0, buff_size)


def write_word(
    word_path, 
    word_params=None, 
    word_name='yi', 
    epi_times=0
):
    """
        write a word and plot
    """
    for index in range(len(word_path)):
        print("*" * 50)
        print("*" * 50)
        print("Write Stroke : {}".format(index))
        stroke_points_index = word_path[index]
 
        if index < (len(word_path) - 1):
            next_index = index + 1
            stroke_points_next_index = word_path[next_index]

            target_angle = np.zeros(2)
            target_angle[0] = stroke_points_next_index[0, 0]
            target_angle[1] = stroke_points_next_index[0, 1]
            stroke_target_point = forward_ik(target_angle)
        else:
            stroke_target_point = Initial_point
        
        write_stroke(
            stroke_points=stroke_points_index,
            stroke_params=word_params[index],
            target_point=stroke_target_point,
            word_name=word_name,
            stroke_name=str(index),
            epi_time=epi_times
        )

        motor_stop()


def write_stroke(
    stroke_points=None,
    stroke_params=None,
    target_point=Initial_point,  
    word_name='yi',   
    stroke_name='0',   
    epi_time=0
    ):
    """
        write a stroke and plot
    """
    way_points = stroke_points
    Num_way_points = way_points.shape[0]

    initial_angle = np.zeros(2)
    initial_angle[0] = way_points[0, 0]
    initial_angle[1] = way_points[0, 1]
    start_point = forward_ik(initial_angle)

    # move to target point
    done = set_pen_up()
    # time.sleep(0.5)
    
    done = move_to_target_point(start_point, Move_Impedance_Params, velocity=0.1)
    # time.sleep(0.5)

    done = set_pen_down()

    time.sleep(0.5)
    
    # params_list = np.tile(impedance_params, (Num_way_points, 1))
    if stroke_params is None:
        exit()
    else:
        params_list = stroke_params

    folder_name = FILE_EVAL_NAME + '/' + word_name
    stroke_angle_name = folder_name + '/' + 'real_angle_list_' + stroke_name + '_' + str(epi_time) + '.txt'
    stroke_torque_name = folder_name + '/' + 'real_torque_list_' + stroke_name + '_' + str(epi_time) + '.txt'
    
    if os.path.exists(folder_name):
        pass
    else:
        os.makedirs(folder_name)

    done = motor_control.run_one_loop(
         way_points[:, 0].copy(), way_points[:, 1].copy(),
         params_list[:, 0].copy(), params_list[:, 1].copy(),
         params_list[:, 2].copy(), params_list[:, 3].copy(),
         Num_way_points,
         Angle_initial[0], Angle_initial[1],
         1,
         stroke_angle_name, stroke_torque_name
    )
    # print("curr_path_list", curr_path_list.shape)
    # np.savetxt('curr_path_list.txt', curr_path_list)
    
    # time.sleep(0.5)

    # move to target point
    done = set_pen_up()
    # time.sleep(0.5)

    done = move_to_target_point(target_point, Move_Impedance_Params, velocity=0.1)

    print("Write stroke once done !!!")
    print("*" * 50)

    return done


def move_to_target_point(
    target_point,
    impedance_params=Move_Impedance_Params,
    velocity=0.05
):
    """
        move to target point
    """
    # done = False

    curr_angle, curr_point = get_observation()
    # dist = np.linalg.norm((curr_point - target_point), ord=2)
    # print("Curr_point (m) :", curr_point)
    # print("Initial dist (m) :", dist)

    angle_list, N = path_planning(curr_point[:2], target_point, velocity=velocity)
    # angle_list = np.loadtxt('angle_list.txt', delimiter=',', skiprows=1)

    N = angle_list.shape[0]

    angle_1_list = angle_list[:, 0].copy()
    angle_2_list = angle_list[:, 1].copy()

    dist_threshold = 0.05
    done = motor_control.move_to_target_point(
        impedance_params[0], impedance_params[1], impedance_params[2], impedance_params[3],
        angle_1_list, angle_2_list, N,
        Angle_initial[0], Angle_initial[1],
        dist_threshold
    )


    # load_eval_path(
    #     root_path='./data/real_path_data',
    #     word_name=None,
    #     epi_times=5
    # )

    # word_path = cope_real_word_path(
    #     root_path='./data/font_data',
    #     word_name='mu',
    #     file_name='real_angle_list_',
    #     epi_times=5,
    #     num_stroke=4,
    #     plot=True
    # )
    # print(trajectory_list.shape)

    # predict_training_samples(
    #     word_name='mu',
    #     stroke_index=0,
    #     re_sample_index=20,
    #     epi_times=5,
    #     num_stroke=4,
    #     plot=True
    # )

    # training_samples_to_waypoints(
    #     word_name='mu'
    # )
