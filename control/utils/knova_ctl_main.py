import numpy as np
import math
import os
import sys
import threading

from path_planning.plot_path import *
from path_planning.path_generate import *
import time
import glob
import scipy

from kortex.api_python.examples import utilities
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2

sns.set(font_scale=1.5)
np.set_printoptions(precision=5)

##############################################################
####################### Hyper Parameters #####################
##############################################################
Move_Impedance_Params = np.array([40.0, 35.0, 4.0, 0.2])
Move_velocity = 0.1
Initial_point = np.array([0.0, 0.0])

##############################################################
########################### kortex ###########################
##############################################################
# Maximum allowed waiting time during actions (in seconds)
TIMEOUT_DURATION = 30


def write_word(word_path, word_params=None, word_name='yi', epi_times=0):
    """
        Write a word and plot :::::
        Input word path in cartesian space ::::
    """
    for index in range(len(word_path)):
        print("*" * 50)
        print("*" * 50)
        # print("Write Stroke %d : " % index)
        stroke_points = word_path[index]

        if index < (len(word_path) - 1):
            stroke_points_next = word_path[index + 1]
            stroke_target_point = np.zeros(2)
            stroke_target_point[0] = stroke_points_next[0, 0]
            stroke_target_point[1] = stroke_points_next[0, 1]
        else:
            stroke_target_point = Initial_point

        # write a stroke
        write_stroke(
            stroke_points=stroke_points,
            stroke_params=word_params[index],
            target_point=stroke_target_point,
            epi_time=epi_times,
            word_name=word_name,
            stroke_name=str(index)
        )


def write_stroke(stroke_points=None,
                 stroke_params=None,
                 target_point=None,   # next initial point
                 epi_time=1,
                 word_name='yi',
                 stroke_name='0'
                 ):
    print("Write stroke !!!%s", word_name + '_' + str(stroke_name))
    done = False

    # obtain prediction value from feedforward model :::
    velocity_list = np.zeros_like(stroke_points)
    preference_force_list = np.zeros_like(stroke_points)

    # params_list = np.tile(impedance_params, (Num_stroke_points, 1)) :::
    if stroke_params is None:
        print("Please give impedance parameters !!!")
        exit()

    start_point = np.zeros(2)
    start_point[0] = stroke_points[0, 0]
    start_point[1] = stroke_points[0, 1]

    done = set_pen_up()
    # time.sleep(0.5)

    # move to target point
    done = move_to_target_point(start_point, Move_Impedance_Params, velocity=0.1)
    # time.sleep(0.5)

    done = set_pen_down()
    # time.sleep(0.5)

    # assistive control for one loop
    done = run_one_loop(
        stroke_points,
        stroke_params,
        velocity_list,
        preference_force_list,
        epi_time,
        word_name,
        stroke_name
    )

    done = set_pen_up()
    # time.sleep(0.5)

    # move to target point
    done = move_to_target_point(target_point, Move_Impedance_Params, velocity=0.1)

    print("Write stroke once done !!!")
    print("*" * 50)

    return done


def set_pen_up():
    """
        pull pen up
    """
    done = False
    time.sleep(1.0)
    return done


def set_pen_down():
    """
        pull pen down
    """
    done = False
    time.sleep(2.0)
    return done


# Create closure to set an event after an END or an ABORT
def check_for_end_or_abort(e):
    """
    Return a closure checking for END or ABORT notifications

    Arguments:
    e -- event to signal when the action is completed
        (will be set when an END or ABORT occurs)
    """
    def check(notification, e=e):
        print("EVENT : " + \
              Base_pb2.ActionEvent.Name(notification.action_event))
        if notification.action_event == Base_pb2.ACTION_END \
                or notification.action_event == Base_pb2.ACTION_ABORT:
            e.set()
    return check


def populateCartesianCoordinate(waypointInformation):
    waypoint = Base_pb2.CartesianWaypoint()

    waypoint.pose.x = waypointInformation[0]
    waypoint.pose.y = waypointInformation[1]
    waypoint.pose.z = waypointInformation[2]
    waypoint.blending_radius = waypointInformation[3]
    waypoint.pose.theta_x = waypointInformation[4]
    waypoint.pose.theta_y = waypointInformation[5]
    waypoint.pose.theta_z = waypointInformation[6]

    waypoint.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
    return waypoint


def move_to_target_point(base,
                         desired_point, desired_impedance_params,
                         velocity=0.1
                         ):
    """
        move to desired point
    """
    # Make sure the arm is in Single Level Servoing mode
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)

    # Move arm to ready position
    print("Moving the arm to the initial position !!!")
    action_type = Base_pb2.RequestedActionType()
    action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
    action_list = base.ReadAllActions(action_type)

    action_handle = None
    for action in action_list.action_list:
        if action.name == "Home":
            action_handle = action.handle

    print("execute action :", action_handle)

    if action_handle == None:
        print("Can't reach safe position. Exiting")
        return False

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    base.ExecuteActionFromReference(action_handle)
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        print("Safe position reached")
    else:
        print("Timeout on action notification wait")

    return finished


def run_one_loop(base,
                 stroke_points, params_list, velocity_list, preference_force_list,
                 eval_epi_times=1,
                 word_name='yi',
                 stroke_name='0'
                 ):
    """
        run one loop
    """
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)
    product = base.GetProductConfiguration()
    waypointsDefinition = tuple(tuple())

    kTheta_x = 90.0
    kTheta_y = 0.0
    kTheta_z = 90.0
    Delta_z = 0.5
    for epi_time in range(eval_epi_times):
        # store data list
        stroke_angle_name = './data/real_path_data/' + word_name + '/' + 'real_angle_list_' + stroke_name + '_' \
                            + str(epi_time) + '.txt'
        stroke_torque_name = './data/real_path_data/' + word_name + '/' + 'real_torque_list_' + stroke_name + '_' \
                             + str(epi_time) + '.txt'

        waypoints = Base_pb2.WaypointList()
        waypoints.duration = 0.0
        waypoints.use_optimal_blending = False

        # send waypoint
        for index in range(stroke_points.shape[0]):
            waypointDefinition = np.array([stroke_points[0], stroke_points[1], Delta_z, 0.0, kTheta_x, kTheta_y, kTheta_z])
            waypoint = waypoints.waypoints.add()
            waypoint.name = "waypoint_" + str(epi_time)
            waypoint.cartesian_waypoint.CopyFrom(populateCartesianCoordinate(waypointDefinition))
            # index = index + 1

        # Verify validity of waypoints
        result = base.ValidateWaypointList(waypoints)
        if (len(result.trajectory_error_report.trajectory_error_elements) == 0):
            e = threading.Event()
            notification_handle = base.OnNotificationActionTopic(
                check_for_end_or_abort(e),
                Base_pb2.NotificationOptions()
            )

            print("Moving cartesian trajectory...")
            base.ExecuteWaypointTrajectory(waypoints)

            print("Waiting for trajectory to finish ...")
            finished = e.wait(TIMEOUT_DURATION)
            base.Unsubscribe(notification_handle)

            if finished:
                print("Cartesian trajectory with no optimization completed ")
                e_opt = threading.Event()
                notification_handle_opt = base.OnNotificationActionTopic(
                    check_for_end_or_abort(e_opt),
                    Base_pb2.NotificationOptions()
                )

                waypoints.use_optimal_blending = True
                base.ExecuteWaypointTrajectory(waypoints)

                print("Waiting for trajectory to finish ...")
                finished_opt = e_opt.wait(TIMEOUT_DURATION)
                base.Unsubscribe(notification_handle_opt)

                if (finished_opt):
                    print("Cartesian trajectory with optimization completed ")
                else:
                    print("Timeout on action notification wait for optimized trajectory")

                return finished_opt
            else:
                print("Timeout on action notification wait for non-optimized trajectory")

            return finished
        else:
            print("Error found in trajectory")
            result.trajectory_error_report.PrintDebugString()

    return done


def load_word_path(root_path='./data/font_data', word_name=None, writing_params=None):
    word_file = root_path + '/' + word_name + '/'
    stroke_list_file = glob.glob(word_file + 'angle_list_*txt')
    print("Load Stroke Data :", len(stroke_list_file))

    word_path = []
    word_params = []
    for i in range(len(stroke_list_file)):
        way_points = np.loadtxt(word_file + 'angle_list_' + str(i) + '.txt', delimiter=' ')

        if writing_params is not None:
            params_list = np.tile(writing_params, (way_points.shape[0], 1))
        else:
            params_list = np.loadtxt(word_file + 'params_list_' + str(i) + '.txt', delimiter=' ')

        N_way_points = way_points.shape[0]
        print("N_way_points :", N_way_points)
        word_path.append(way_points.copy())
        word_params.append(params_list.copy())
    return word_path, word_params


def load_real_path(root_path='./data/font_data', word_name=None):
    word_file = root_path + '/' + word_name + '/'
    stroke_list_file = glob.glob(word_file + 'real_angle_list_*txt')
    print("Load Stroke Data :", len(stroke_list_file))
    real_path = []
    for i in range(len(stroke_list_file)):
        real_way_points = np.loadtxt(root_path + 'real_angle_list_' + str(i) + '.txt', delimiter=' ', skiprows=1)
        real_path.append(real_way_points)

        # truth_data = scipy.signal.resample(real_way_points, 100)
        # down_sample = scipy.signal.resample(real_way_points, 100)
        # down_sample = real_way_points

        # idx = np.arange(0, down_sample.shape[0], down_sample.shape[0]/100).astype('int64')
        # real_angle_list = np.zeros((idx.shape[0], 2))
        # desired_angle_list = np.zeros((idx.shape[0], 2))
        # real_angle_list[:, 0] = down_sample[idx, 1]
        # real_angle_list[:, 1] = down_sample[idx, 4]
        # real_angle_list[:, 0] = scipy.signal.resample(down_sample[:, 1], 100)
        # real_angle_list[:, 1] = scipy.signal.resample(down_sample[:, 4], 100)

        # real_2d_path = forward_ik_path(real_angle_list)
        # print(real_2d_path)
        # real_path.append(real_2d_path.copy())
        # np.array(real_path).reshape(np.array(real_path).shape[0] * np.array(real_path).shape[1], 2)

    return np.array(real_path)


def start_api():
    # Parse arguments
    args = utilities.parseConnectionArguments()

    # Create connection to the device and get the router
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        # Create required services
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)

    return base, base_cyclic


if __name__ == "__main__":
    # word_path, word_params = load_word_path(
    #     root_path='./data/font_data',
    #     word_name='mu',
    #     joint_params=np.array([1.0, 1.0])
    # )
    x_list, y_list = plot_real_word_2d_path(
        root_path='./data/font_data',
        word_name='mu',
        stroke_num=4,
        delimiter=' ',
        skiprows=0,
    )

    osc_data_list = transform_task_space(x_list, y_list, offset=np.array([0.0, 0.0]))

    plot_real_osc_2d_path(
        osc_data_list
    )
