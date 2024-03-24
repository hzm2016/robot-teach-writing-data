import cv2
import sys
import os
from control.vision_capture.main_functions import * 
from control.path_planning.path_generate import * 
from control.path_planning.plot_path import * 
import socket 
import time 
from control.protocol.task_interface import * 
from control.protocol.config import config


def eval_writing(run_on=True, Load_path=False):
	"""
		eval writting performance :
	"""
	
	_server = Server(5005)
	
	# ######################################################
	# ############## wait encoder and motor check ##########
	# ################### Position calibrate ###############
	# ######################################################
	_server.wait_encoder_request()
	curr_angle, curr_point = get_observation(Angle_initial)
	_server.send_encoder_check(curr_point)
	
	if not Load_path:
		print("Load stroke path !!!")
		stroke_angle = np.loadtxt('angle_list_0.txt', delimiter=' ')
	# N_way_points = stroke_angle.shape[0]
	# print("N_way_points :", N_way_points)
	
	# ######################################################
	# ############## Wait impedance parameters  ############
	# ######################################################
	_server.wait_params_request()
	
	# impedance_params = None
	# while impedance_params is None:
	# read impedance parameters :::
	while True:
		impedance_params = _server.read_params()
		impedance_params = np.array(impedance_params.copy())
		
		if impedance_params is np.NaN:
			exit()
		
		if impedance_params is not None:
			break
	
	time.sleep(1.0)
	# impedance_params = np.array([35.0, 24.0, 0.0, 0.0])
	print("Input impedance parameters :::", np.array(impedance_params))
	print("+" * 50)
	
	num_eval = 3
	for i in range(num_eval):
		print('Writting episode %d:' % i)
		if run_on:
			write_stroke(stroke_points=stroke_angle, impedance_params=np.array([35.0, 30.0, 1.4, 0.2]),
			             target_point=Initial_point)
			
			print("*" * 50)
			print("Eval one stroke once done !!!")
		
		# send movement_done command
		_server.send_movement_done()
	
	motor_control.motor_3_stop()
	_server.close()
	

def server_train(angle_initial=Angle_initial, run_on=True, Load_path=False):
	_server = Server(5005)
	
	# ######################################################
	# ############## wait encoder and motor check ##########
	# ################### Position calibrate ###############
	# ######################################################
	_server.wait_encoder_request()
	curr_angle, curr_point = get_observation(angle_initial)
	_server.send_encoder_check(angle_initial)
	
	# move_to_target_point(Initial_angle)
	
	# ######################################################
	# ############## Wait way_points #######################
	# ######################################################
	_server.wait_way_points_request()
	
	print("+" * 50)
	# receive way points
	way_points = []
	
	if Load_path:
		os.remove(r'data/real_path_data/angle_list.txt')
		data_file = open('data/real_path_data/angle_list.txt', 'w')
	
	way_point = None
	while way_point != "SEND_DONE":
		way_point = _server.read_way_points()
		# print("way_points ::::", way_point)
		if way_point == "SEND_DONE":
			break
		way_points.append(way_point.copy())
		
		line_data = str(way_point[0]) + ',' + str(way_point[1]) + '\n'
		
		if Load_path:
			data_file.writelines(line_data)
	# send_done = _server.wait_send_way_points_done()
	
	way_points = np.array(way_points)
	N_way_points = way_points.shape[0]
	# print("way_points :::", way_points.shape)
	print("N_way_points :::", N_way_points)
	print("+" * 50)
	
	# ######################################################
	# ############## Wait impedance parameters  ############
	# ######################################################
	_server.wait_params_request()
	
	# impedance_params = None
	# while impedance_params is None:
	# read impedance parameters :::
	impedance_params = _server.read_params()
	impedance_params = np.array(impedance_params.copy())
	print("Input impedance parameters :::", np.array(impedance_params))
	print("+" * 50)
	if impedance_params is np.NaN:
		exit()
	
	time.sleep(2.0)
	impedance_params = np.array([35.0, 24.0, 0.0, 0.0])
	
	# start move
	if not Load_path:
		# start_point = forward_ik(way_points[0, :].copy())
		# print("Move to start point :::", start_point)
		way_points = np.loadtxt('angle_list.txt', delimiter=',')
		N_way_points = way_points.shape[0]
	
	if run_on:
		initial_angle = np.zeros(2)
		initial_angle[0] = way_points[0, 0]
		initial_angle[1] = way_points[0, 1]
		start_point = forward_ik(initial_angle)
		move_impedance_params = np.array([20.0, 16.0, 0.1, 0.1])
		
		move_to_target_point(start_point, move_impedance_params, velocity=0.05)
		
		motor_control.run_one_loop(impedance_params[0], impedance_params[1], impedance_params[2], impedance_params[3],
		                           way_points[:, 0].copy(), way_points[:, 1].copy(), N_way_points,
		                           Angle_initial[0], Angle_initial[1], 1)
		
		time.sleep(2.0)
		move_to_target_point(Initial_point, move_impedance_params, velocity=0.05)
	
	# send movement_done command
	_server.send_movement_done()
	
	_server.close()


def run_main(show_video=False):
	# initial TCP connection :::
	# check encoders and motors :::
	task = TCPTask('169.254.0.99', 5005)
	
	# check motor and encode well before experiments
	# task.get_encoder_check()
	task.send_params_request()
	
	# # generate impedance parameters::
	# GP-UCB
	# ================================================================
	
	
	# ================================================================
	
	# # send impedance params :::
	stiffness = [100, 100]
	damping = [50, 50]
	params = stiffness + damping
	task.send_params(params)
	
	# offline check the generated path :::
	angle_1_list_e, angle_2_list_e = check_path(root_path='path_planning/font_data', plot_show=False, font_name='third', type=3)
	way_points = np.vstack((angle_1_list_e, angle_2_list_e)).transpose()
	print("way_points :::", way_points.shape)

	task.send_way_points_request()
	
	task.send_way_points(way_points)
	
	task.send_way_points_done()
	
	# # send way_points :::
	# command_move = "Move_start"
	
	# if show_video:
	# 	show_video()
		
	# video record for trail :::
	run_done = task.get_movement_check()
	if run_done:
		print("run_done", run_done)
	
	# if run_done:
	# 	capture_image(root_path='captured_images/', font_name='test')
	# 	image_precessing(img_path='captured_images/', img_name='test')
	

if __name__ == "__main__":
	root_path = 'data/font_data'
	font_name = 'yi'
	type = 1
	# path_data = np.loadtxt(root_path + '/' + font_name + '/1_font_' + str(type) + '.txt')
	# way_points = generate_path(path_data, center_shift=np.array([0.16, -WIDTH/2]), velocity=10, Ts=0.001, plot_show=True)
	
	# plot_real_2d_path(
	# 	root_path='',
	# 	file_name='angle_list.txt'
	# ) 
	
	# check_path(root_path='data/font_data', font_name='third',
	#            type=3, period=10, Ts=0.001)
	
	# capture_image(root_path='data/captured_images/', font_name='test_image')

	# show_video()
#
# 	run_main()
	
	# n_clusters = config["reacher2d_1"]["n_cluster"]
	#
	# task = config["reacher2d_1"]["task_box"](False)
	# print("tasks group :::", task._group)
	#
	# weights = np.random.rand(40).tolist()
	# results = task.get_movement(weights=weights, duration=10.)
	# print("results :::", results)