#! /usr/bin/env python
##################################################
# Author: Noemie Jaquier, 2019
# License: MIT
# Contact: noemie.jaquier@idiap.ch
##################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat  # loading data from matlab
from forward_mode.utils.gmr import Gmr, plot_gmm

# GMR on 2D trajectories with time as input
if __name__ == '__main__':
	# Load data
	file_name = './forward_mode'
	letter = 'B'
	datapath = file_name + '/data/2Dletters/'
	data = loadmat(datapath + '%s.mat' % letter)
	# print("data :", data.shape)
	demos = [d['pos'][0][0].T for d in data['demos'][0]]
	# print("demos :", demos.shape)

	# Parameters
	nb_data = demos[0].shape[0]
	print("nb_data :", nb_data)
	nb_data_sup = 50
	nb_samples = 5
	dt = 0.01
	input_dim = 1
	output_dim = 2
	in_idx = [0]
	out_idx = [1, 2]
	nb_states = 6

	# Create time data
	demos_t = [np.arange(demos[i].shape[0])[:, None] for i in range(nb_samples)]
	# print("demos_t :", demos_t)

	# Stack time and position data
	demos_tx = [np.hstack([demos_t[i]*dt, demos[i]]) for i in range(nb_samples)]
	# print("demos_tx :", demos_tx)

	# Stack demos
	demos_np = demos_tx[0]
	for i in range(1, nb_samples):
		demos_np = np.vstack([demos_np, demos_tx[i]])
	print("demos_np :", demos_np.shape)

	X = demos_np[:, 0][:, None]
	Y = demos_np[:, 1:]
	# print('Y :', Y)
	print('Y shape:', Y.shape)

	# Test data
	Xt = dt * np.arange(nb_data + nb_data_sup)[:, None]

	# GMM
	gmr_model = Gmr(nb_states=nb_states, nb_dim=input_dim+output_dim, in_idx=in_idx, out_idx=out_idx)
	gmr_model.init_params_kbins(demos_np.T, nb_samples=nb_samples)
	gmr_model.gmm_em(demos_np.T)

	# GMR
	mu_gmr = []
	sigma_gmr = []
	sigma_gmr_1 = []
	sigma_gmr_2 = []
	for i in range(Xt.shape[0]):
		mu_gmr_tmp, sigma_gmr_tmp, H_tmp = gmr_model.gmr_predict(Xt[i])
		mu_gmr.append(mu_gmr_tmp)
		sigma_gmr.append(sigma_gmr_tmp)
		sigma_gmr_1.append(sigma_gmr_tmp[0, 0])
		sigma_gmr_2.append(sigma_gmr_tmp[1, 1])
		print("mu_gmr :", mu_gmr_tmp.shape)
		print("sigma_gmr :", sigma_gmr_tmp.shape)

	sigma_gmr_1_diag = np.diag(np.array(sigma_gmr_1))
	sigma_gmr_2_diag = np.diag(np.array(sigma_gmr_2))
	mu_gmr = np.array(mu_gmr)
	sigma_gmr = np.array(sigma_gmr)

	# Plots
	plt.figure(figsize=(5, 5))
	for p in range(nb_samples):
		plt.plot(Y[p * nb_data:(p + 1) * nb_data, 0], Y[p * nb_data:(p + 1) * nb_data, 1], color=[.7, .7, .7])
		plt.plot(Y[p * nb_data, 0], Y[p * nb_data, 1], color=[.7, .7, .7], marker='o')
	plot_gmm(np.array(gmr_model.mu)[:, 1:], np.array(gmr_model.sigma)[:, 1:, 1:], alpha=0.6, color=[0.1, 0.34, 0.73])
	axes = plt.gca()
	axes.set_xlim([-14., 14.])
	axes.set_ylim([-14., 14.])
	plt.xlabel('$y_1$', fontsize=30)
	plt.ylabel('$y_2$', fontsize=30)
	plt.locator_params(nbins=3)
	plt.tick_params(labelsize=20)
	plt.tight_layout()
	plt.savefig(file_name + '/figures/GMM_B.png')

	plt.figure(figsize=(5, 5))
	for p in range(nb_samples):
		plt.plot(Y[p*nb_data:(p+1)*nb_data, 0], Y[p*nb_data:(p+1)*nb_data, 1], color=[.7, .7, .7])
		plt.scatter(Y[p*nb_data, 0], Y[p*nb_data, 1], color=[.7, .7, .7], marker='X', s=80)
	plt.plot(mu_gmr[:, 0], mu_gmr[:, 1], color=[0.20, 0.54, 0.93], linewidth=3)
	plt.scatter(mu_gmr[0, 0], mu_gmr[0, 1], color=[0.20, 0.54, 0.93], marker='X', s=80)
	plot_gmm(mu_gmr, sigma_gmr, alpha=0.05, color=[0.20, 0.54, 0.93])

	print("sigma_mr :", sigma_gmr.shape)
	mu_gmr_sample_1 = []
	mu_gmr_sample_2 = []
	for i in range(1):
		prior_traj_sample_1 = np.random.multivariate_normal(mu_gmr[:, 0], sigma_gmr_1_diag)
		prior_traj_sample_2 = np.random.multivariate_normal(mu_gmr[:, 1], sigma_gmr_2_diag)
		# prior_traj_sample = np.random.multivariate_normal(mu_gmr[:, :], sigma_gmr_1_diag)
	# for j in range(Xt.shape[0]):
		# 	prior_traj_sample = np.random.multivariate_normal(mu_gmr[j, :], sigma_gmr[j, :, :])
		# 	mu_gmr_sample.append(prior_traj_sample)
	# print(np.array(mu_gmr_sample).shape)
	plt.plot(prior_traj_sample_1, prior_traj_sample_2, color=[0.93, 0.20, 0.54], linewidth=3)

	axes = plt.gca()
	axes.set_xlim([-14, 14.])
	axes.set_ylim([-14., 14.])
	plt.xlabel('$y_1$', fontsize=30)
	plt.ylabel('$y_2$', fontsize=30)
	plt.locator_params(nbins=3)
	plt.tick_params(labelsize=20)
	plt.tight_layout()
	plt.savefig(file_name + '/figures/GMR_B.png')

	plt.figure(figsize=(5, 4))
	for p in range(nb_samples):
		plt.plot(Xt[:nb_data, 0], Y[p * nb_data:(p + 1) * nb_data, 0], color=[.7, .7, .7])
	plt.plot(Xt[:, 0], mu_gmr[:, 0], color=[0.20, 0.54, 0.93], linewidth=3)
	plt.plot(Xt[:, 0], prior_traj_sample_1, color=[0.20, 0.54, 0.93], linewidth=3)
	miny = mu_gmr[:, 0] - np.sqrt(sigma_gmr[:, 0, 0])
	maxy = mu_gmr[:, 0] + np.sqrt(sigma_gmr[:, 0, 0])
	plt.fill_between(Xt[:, 0], miny, maxy, color=[0.20, 0.54, 0.93], alpha=0.3)
	prior_traj = []

	axes = plt.gca()
	axes.set_ylim([-14., 14.])
	plt.xlabel('$t$', fontsize=30)
	plt.ylabel('$y_1$', fontsize=30)
	plt.tick_params(labelsize=20)
	plt.tight_layout()
	plt.savefig(file_name + '/figures/GMR_B01.png')

	plt.figure(figsize=(5, 4))
	for p in range(nb_samples):
		plt.plot(Xt[:nb_data, 0], Y[p * nb_data:(p + 1) * nb_data, 1], color=[.7, .7, .7])
	plt.plot(Xt[:, 0], mu_gmr[:, 1], color=[0.20, 0.54, 0.93], linewidth=3)
	plt.plot(Xt[:, 0], prior_traj_sample_2, color=[0.20, 0.54, 0.93], linewidth=3)
	miny = mu_gmr[:, 1] - np.sqrt(sigma_gmr[:, 1, 1])
	maxy = mu_gmr[:, 1] + np.sqrt(sigma_gmr[:, 1, 1])
	plt.fill_between(Xt[:, 0], miny, maxy, color=[0.20, 0.54, 0.93], alpha=0.3)
	axes = plt.gca()
	axes.set_ylim([-14., 14.])
	plt.xlabel('$t$', fontsize=30)
	plt.ylabel('$y_2$', fontsize=30)
	plt.tick_params(labelsize=20)
	plt.tight_layout()
	plt.savefig(file_name + '/figures/GMR_B02.png')
	plt.show()
