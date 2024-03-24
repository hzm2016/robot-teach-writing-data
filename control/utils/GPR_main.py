import numpy as np
import math
import os
from path_planning.plot_path import *
from path_planning.path_generate import *
from path_planning.utils import *
import time
import glob
import scipy
import matplotlib.pyplot as plt
from scipy.io import loadmat  # loading data from matlab
from forward_mode.utils.gmr import Gmr, plot_gmm
from forward_mode.utils.gp_coregionalize_with_mean_regression import GPCoregionalizedWithMeanRegression
from forward_mode.utils.gmr_mean_mapping import GmrMeanMapping
from forward_mode.utils.gmr_kernels import Gmr_based_kernel
import GPy
from scipy import signal

sns.set(font_scale=1.5)
np.set_printoptions(precision=5)

if __name__ == "__main__":
	file_fig_name = './data/predicted_images/'
	write_name = 'yi_10'
	stroke_num = 1
	stroke_index = 0
	epi_times = 5
	dt = 0.001
	nb_samples = 5
	re_sample_index = 20

	input_dim = 1
	output_dim = 2
	in_idx = [0]
	out_idx = [1, 2]
	nb_states = 5

	nb_data_sup = 10

	# plot font size
	font_size = 20

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
	print("x_list :", np.array(x_list).shape, "y_list :", np.array(y_list).shape)
	
	folder_name = './data/predicted_images/' + write_name
	if os.path.exists(folder_name):
		pass
	else:
		os.makedirs(folder_name)
	
	# down sampling ::::
	# x_down_list = signal.resample(x_list, 200, axis=1)
	# y_down_list = signal.resample(y_list, 200, axis=1)
	x_down_list = []
	y_down_list = []
	for i in range(len(x_list)):
		x_down_list.append(x_list[i][::re_sample_index])
		y_down_list.append(y_list[i][::re_sample_index])
	print('x_list shape :', np.array(x_down_list).shape, 'y list shape :', np.array(y_down_list).shape)

	nb_data = x_down_list[0].shape[0]
	print("nb_data :", nb_data)
	
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
	print("Y :", Y.shape)

	X_list = [X for i in range(output_dim)]
	Y_list = [Y[:, i][:, None] for i in range(output_dim)]

	# Test data
	Xt = dt * np.arange(nb_data + nb_data_sup)[:, None]
	nb_data_test = Xt.shape[0]
	Xtest, _, output_index = GPy.util.multioutput.build_XY([Xt for i in range(output_dim)])

	# Create coregionalisation model
	kernel = GPy.kern.Matern52(input_dim, variance=1., lengthscale=1.)
	K = kernel.prod(GPy.kern.Coregionalize(1, output_dim, active_dims=[input_dim], name='B'))
	m = GPy.models.GPCoregionalizedRegression(X_list=X_list, Y_list=Y_list)
	m.randomize()
	m.optimize('bfgs', max_iters=100, messages=True)

	# Prediction
	mu_gp, sigma_gp = m.predict(Xtest, full_cov=True, Y_metadata={'output_index': output_index})
	mu_gp_rshp = np.reshape(mu_gp, (output_dim, -1))

	sigma_gp_tmp = np.zeros((nb_data_test, nb_data_test, output_dim * output_dim))
	for i in range(output_dim):
		for j in range(output_dim):
			sigma_gp_tmp[:, :, i * output_dim + j] = sigma_gp[i * nb_data_test:(i + 1) * nb_data_test,
													 j * nb_data_test:(j + 1) * nb_data_test]
	sigma_gp_rshp = np.zeros((nb_data_test, output_dim, output_dim))
	for i in range(nb_data_test):
		sigma_gp_rshp[i] = np.reshape(sigma_gp_tmp[i, i, :], (2, 2))

	# Plots
	plt.figure(figsize=(5, 5))
	for p in range(nb_samples):
		plt.plot(Y[p * nb_data:(p + 1) * nb_data, 0], Y[p * nb_data:(p + 1) * nb_data, 1], color=[.7, .7, .7])
		plt.scatter(Y[p * nb_data, 0], Y[p * nb_data, 1], color=[.7, .7, .7], marker='X', s=80)
	plot_gmm(mu_gp_rshp.T, sigma_gp_rshp, alpha=0.05, color=[0.99, 0.76, 0.53])
	plt.plot(mu_gp_rshp[0, :], mu_gp_rshp[1, :], color=[0.99, 0.76, 0.53], linewidth=3)
	plt.scatter(mu_gp_rshp[0, 0], mu_gp_rshp[1, 0], color=[0.99, 0.76, 0.53], marker='X', s=80)
	axes = plt.gca()
	# axes.set_xlim([-14., 14.])
	# axes.set_ylim([-14., 14.])
	axes.set_xlim([0.1, 0.6])
	axes.set_ylim([-0.25, 0.25])
	plt.xlabel('$x_1(m)$', fontsize=30)
	plt.ylabel('$x_2(m)$', fontsize=30)
	plt.locator_params(nbins=3)
	plt.tick_params(labelsize=20)
	plt.tight_layout()
	# plt.savefig('figures/GP_B.png')

	# plt.figure(figsize=(5, 4))
	# for p in range(nb_samples):
	# 	plt.plot(Xt[:nb_data, 0], Y[p * nb_data:(p + 1) * nb_data, 0], color=[.7, .7, .7])
	# plt.plot(Xt[:, 0], mu_gp_rshp[0, :], color=[0.99, 0.76, 0.53], linewidth=3)
	# miny = mu_gp_rshp[0, :] - np.sqrt(sigma_gp_rshp[:, 0, 0])
	# maxy = mu_gp_rshp[0, :] + np.sqrt(sigma_gp_rshp[:, 0, 0])
	# plt.fill_between(Xt[:, 0], miny, maxy, color=[0.99, 0.76, 0.53], alpha=0.3)
	# axes = plt.gca()
	# # axes.set_ylim([-14., 14.])
	# plt.xlabel('$t$', fontsize=30)
	# plt.ylabel('$y_1$', fontsize=30)
	# plt.tick_params(labelsize=20)
	# plt.tight_layout()
	# # plt.savefig('figures/GP_B01.png')

	# plt.figure(figsize=(5, 4))
	# for p in range(nb_samples):
	# 	plt.plot(Xt[:nb_data, 0], Y[p * nb_data:(p + 1) * nb_data, 1], color=[.7, .7, .7])
	# plt.plot(Xt[:, 0], mu_gp_rshp[1, :], color=[0.99, 0.76, 0.53], linewidth=3)
	# miny = mu_gp_rshp[1, :] - np.sqrt(sigma_gp_rshp[:, 1, 1])
	# maxy = mu_gp_rshp[1, :] + np.sqrt(sigma_gp_rshp[:, 1, 1])
	# plt.fill_between(Xt[:, 0], miny, maxy, color=[0.99, 0.76, 0.53], alpha=0.3)
	# axes = plt.gca()
	# # axes.set_ylim([-14., 14.])
	# plt.xlabel('$t$', fontsize=30)
	# plt.ylabel('$y_2$', fontsize=30)
	# plt.tick_params(labelsize=20)
	# plt.tight_layout()
	# # plt.savefig('figures/GP_B02.png')
	# #
	# # plt.show()

	# New observations (via-points to go through)
	# X_obs = np.array([0.0, 0.15, 0.21])[:, None]
	# Y_obs = np.array([[0.3, -0.02], [0.40, -0.10], [0.42, -0.17]])
	# X_obs = np.array([0.05, 0.22])[:, None]
	# Y_obs = np.array([[0.28, -0.10], [0.26, 0.06]])
	# X_obs = np.array([0.05, 0.12, 0.22])[:, None]
	X_obs = np.array([0.01, 0.08, 0.176])[:, None]
	Y_obs = np.array([[0.32, -0.15], [0.3, -0.03], [0.35, 0.16]])
	X_obs_list = [np.hstack((X_obs, X_obs)) for i in range(output_dim)]
	Y_obs_list = [Y_obs[:, i][:, None] for i in range(output_dim)]
	Xobstest, _, output_index_obs = GPy.util.multioutput.build_XY([X_obs for i in range(output_dim)])
	nb_obs = X_obs.shape[0]

	# Test data
	Xt = dt * np.arange(nb_data + nb_data_sup)[:, None]
	nb_data_test = Xt.shape[0]
	Xtest, _, output_index = GPy.util.multioutput.build_XY([Xt for i in range(output_dim)])

	# Create coregionalisation model
	# kernel = GPy.kern.Matern52(input_dim, variance=0.01, lengthscale=0.01)
	kernel = GPy.kern.RBF(input_dim, variance=0.001, lengthscale=0.1)
	K = kernel.prod(GPy.kern.Coregionalize(1, output_dim, active_dims=[input_dim], name='B'))
	m = GPy.models.GPCoregionalizedRegression(X_list=X_list, Y_list=Y_list)
	m.randomize()
	m.optimize('bfgs', max_iters=100, messages=True)

	# Prediction
	mu_gp_test, sigma_gp_test = m.predict(Xtest, full_cov=True, Y_metadata={'output_index': output_index})
	mu_gp_obs, sigma_gp_obs = m.predict(Xobstest, full_cov=True, Y_metadata={'output_index': output_index_obs})

	# Trajectory modulation
	k_test = K.K(Xtest)
	k_test_obs = K.K(Xtest, Xobstest)
	k_obs = K.K(Xobstest)
	mu_gp = mu_gp_test + np.dot(np.dot(k_test_obs, np.linalg.inv(k_obs)), Y_obs.reshape((nb_obs*output_dim, 1), order='F') - mu_gp_obs)
	sigma_gp = k_test + np.dot(np.dot(k_test_obs, np.linalg.inv(k_obs)), k_test_obs.T)

	mu_gp_rshp = np.reshape(mu_gp, (output_dim, -1))
	sigma_gp_tmp = np.zeros((nb_data_test, nb_data_test, output_dim*output_dim))
	for i in range(output_dim):
		for j in range(output_dim):
			sigma_gp_tmp[:, :, i*output_dim+j] = sigma_gp[i*nb_data_test:(i+1)*nb_data_test, j*nb_data_test:(j+1)*nb_data_test]
	sigma_gp_rshp = np.zeros((nb_data_test, output_dim, output_dim))
	for i in range(nb_data_test):
		sigma_gp_rshp[i] = np.reshape(sigma_gp_tmp[i, i, :], (2, 2))

	# Plots
	plt.figure(figsize=(5, 5))
	for p in range(nb_samples):
		plt.plot(Y[p*nb_data:(p+1)*nb_data, 0], Y[p*nb_data:(p+1)*nb_data, 1], color=[.7, .7, .7])
		plt.plot(Y[p*nb_data, 0], Y[p*nb_data, 1], color=[.7, .7, .7], marker='o')
	plot_gmm(mu_gp_rshp.T, sigma_gp_rshp, alpha=0.05, color=[0.99, 0.76, 0.53])
	plt.plot(mu_gp_rshp[0, :], mu_gp_rshp[1, :], color=[0.99, 0.76, 0.53], linewidth=3)
	plt.plot(mu_gp_rshp[0, 0], mu_gp_rshp[1, 0], color=[0.99, 0.76, 0.53], marker='o')
	plt.scatter(Y_obs[:, 0], Y_obs[:, 1], color=[0, 0, 0], zorder=60, s=100)
	axes = plt.gca()
	# axes.set_xlim([-17., 17.])
	# axes.set_ylim([-17., 17.])
	axes.set_xlim([0.1, 0.6])
	axes.set_ylim([-0.25, 0.25])
	plt.xlabel('$x_1(m)$', fontsize=30)
	plt.ylabel('$x_2(m)$', fontsize=30)
	plt.locator_params(nbins=3)
	plt.tick_params(labelsize=20)
	plt.tight_layout()
	# plt.savefig('figures/GP_B_trajmod.png')

	# plt.figure(figsize=(5, 4))
	# for p in range(nb_samples):
	# 	plt.plot(Xt[:nb_data, 0], Y[p*nb_data:(p+1)*nb_data, 0], color=[.7, .7, .7])
	# plt.plot(Xt[:, 0], mu_gp_rshp[0, :], color=[0.99, 0.76, 0.53], linewidth=3)
	# miny = mu_gp_rshp[0, :] - np.sqrt(sigma_gp_rshp[:, 0, 0])
	# maxy = mu_gp_rshp[0, :] + np.sqrt(sigma_gp_rshp[:, 0, 0])
	# plt.fill_between(Xt[:, 0], miny, maxy, color=[0.99, 0.76, 0.53], alpha=0.3)
	# plt.scatter(X_obs[:, 0], Y_obs[:, 0], color=[0, 0, 0], zorder=60, s=100)
	# axes = plt.gca()
	# # axes.set_ylim([-17., 17.])
	# plt.xlabel('$t$', fontsize=30)
	# plt.ylabel('$y_1$', fontsize=30)
	# plt.tick_params(labelsize=20)
	# plt.tight_layout()
	# # plt.savefig('figures/GP_B01_trajmod.png')

	# plt.figure(figsize=(5, 4))
	# for p in range(nb_samples):
	# 	plt.plot(Xt[:nb_data, 0], Y[p*nb_data:(p+1)*nb_data, 1], color=[.7, .7, .7])
	# plt.plot(Xt[:, 0], mu_gp_rshp[1, :], color=[0.99, 0.76, 0.53], linewidth=3)
	# miny = mu_gp_rshp[1, :] - np.sqrt(sigma_gp_rshp[:, 1, 1])
	# maxy = mu_gp_rshp[1, :] + np.sqrt(sigma_gp_rshp[:, 1, 1])
	# plt.fill_between(Xt[:, 0], miny, maxy, color=[0.99, 0.76, 0.53], alpha=0.3)
	# plt.scatter(X_obs[:, 0], Y_obs[:, 1], color=[0, 0, 0], zorder=60, s=100)
	# axes = plt.gca()
	# # axes.set_ylim([-17., 17.])
	# plt.xlabel('$t$', fontsize=30)
	# plt.ylabel('$y_1$', fontsize=30)
	# plt.tick_params(labelsize=20)
	# plt.tight_layout()

	plt.show()