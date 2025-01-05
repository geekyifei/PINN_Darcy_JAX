#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 14:09:03 2024

@author: yifei_linux
"""

# Import dependencies
import optax
import pickle
import os
import h5py
import jax
import jax.numpy as jnp
from jax import grad, vmap, jit, devices
from jax import random
from jax.flatten_util import ravel_pytree
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
import utils as utils
#from PINNDarcy2D_dynamic_weight import PinnDarcy2D
from PINNDarcy2D import PinnDarcy2D

print(f"JAX is using: {devices()[0]}")
device = jax.devices("gpu")[0]
rand_seed = 111
np.random.seed(rand_seed)
key = random.PRNGKey(rand_seed)
key, subkey = random.split(key, 2)
dtype = np.float32

# Load data
with h5py.File('./2d_ade_var1_corlen01_ptsrc_e0025_M1.h5', 'r') as f:
    # print([key for key in f.keys()])
    k_ref = f.get('k_ref')[:]  # (Nx*Ny,)
    h_ref = f.get('h_ref')[:]  # (Nx*Ny,)
    x_ref = f.get('x_ref')[:]  # (Nx*Ny, 2), cell centers!
y_ref = np.log(k_ref)
norm_y = max(y_ref.min(), y_ref.max(), key=abs)
norm_h = max(h_ref.min(), h_ref.max(), key=abs)

# Domain geometry (x1, x2)
lbt = np.array([0., 0.])  # domain lower corner
ubt = np.array([1., 0.5])  # domain upper corner
Nx1 = 256  # cells
Nx2 = 128  # cells
N = Nx1*Nx2
x1 = np.linspace(x_ref[0, 0], x_ref[-1, 0], Nx1)  # cell centers
x2 = np.linspace(x_ref[0, 1], x_ref[-1, 1], Nx2)  # cell centers
XX1, XX2 = np.meshgrid(x1, x2)
bc_left_dar = 1  # Dirichlet BC
bc_right_dar = 0 # Dirichlet BC
bc_top_dar = 0  # No-flow
bc_bot_dar = 0  # No-flow

# PINN hyperparameters
Ny = 200
Nh = 40
Nres_dar = 5000  # 500
Nbl = Nbr = 16  # 32
Nbt = Nbb = 32  # 64

lambda_res_dar = 1
lambda_bl = 1
lambda_br = 1
lambda_bt = 1
lambda_bb = 1
lambda_y = 1
lambda_h = 1
lambda_p = 0

# IMPORTANT FLAGs
train_mode = 'inverse'  # valid keys 'inverse', 'regression', 'forward'
if_load_weights = False  # FLAG for loading trained DNNs for y and h
if_fourier_feature = True  # FLAG for using Fourier feature DNNs for y
if_normalize_data = True  # FLAG for applying normalization to data
if_obs_loc_same = True  # FLAG for if k and h obs locations are the same
if_lbfgs = False  # FLAG for if using lbfgs optimizer after Adam
if_full_batch = True  # FLAG for if using full-batch training for Adam

# PINN training
hidden_size = 60
if if_fourier_feature:
    layers_y = [2*hidden_size, 4*hidden_size, 4*hidden_size, 2*hidden_size, 1]
else:
    layers_y = [2, hidden_size, hidden_size, hidden_size, hidden_size, 1]
layers_h = [2, hidden_size, hidden_size, hidden_size, hidden_size, 1]

num_epoch_y = 20000
num_epoch_h = 20000
num_epoch_dar = 50000
num_print = 200
batch_size = 500
lbfgs_max = 20000
lbfgs_rtol = 1e-8

path_f = f'./Darcy_{train_mode}_Ny_{Ny}_Nh_{Nh}_Nr_{Nres_dar}'
path_fig = os.path.join(path_f, 'figures')
if not os.path.exists(path_f):
    os.makedirs(path_f)
if not os.path.exists(path_fig):
    os.makedirs(path_fig)
f_rec = open(os.path.join(path_f, 'record.out'), 'a+')  # Record File

# Create a list of variable names to include in the params dictionary
variable_names = [
    'lbt', 'ubt', 'Nx', 'Ny', 'N',
    'Ny', 'Nh', 'Nc', 'Nres_dar', 'Nbl', 'Nbr', 'Nbt', 'Nbb',
    'lambda_res_dar', 'lambda_bl', 'lambda_br', 'lambda_bt',
    'lambda_bb', 'lambda_y', 'lambda_h', 'lambda_p',
    'rand_seed', 'num_epoch_y', 'num_epoch_h', 'num_epoch_dar', 'num_print',
    'train_mode', 'if_fourier_feature', 'if_load_weights', 'if_normalize_data',
    'if_obs_loc_same', 'if_lbfgs', 'lbfgs_max', 'lbfgs_rtol'
]
params_dict = utils.ParamsDict(globals())
params_dict.add_items(variable_names)
for key_, value in params_dict.params.items():
    print(f"{key_}: {value} \n", file=f_rec)
params = params_dict.params

# =============================================================================
# Prepare training datasets
# =============================================================================

# Create dataset
idx_y = np.random.choice(N, Ny, replace=False)
idx_h = idx_y if if_obs_loc_same else np.random.choice(N, Nh, replace=False)
remaining_indices = np.setdiff1d(np.arange(N), idx_y)
idx_test = np.random.choice(remaining_indices, int(Ny/2), replace=False)

dataset = dict()
x_y = x_ref[idx_y, :]
y_y = y_ref[idx_y][:, np.newaxis]
# norm_y = max(y_y.min(), y_y.max(), key=abs) if (Ny !=0 & if_normalize_data) else 1
y_data = jnp.concatenate([x_y, y_y], axis=1)
y_test = jnp.concatenate(
    [x_ref[idx_test, :], y_ref[idx_test][:, np.newaxis]], axis=1)
dataset.update({'y_data': y_data})

h_ref_rs = h_ref.reshape(Nx2, Nx1)
x_h = x_ref[idx_h, :]
y_h = h_ref[idx_h][:, np.newaxis]
# norm_h = max(y_h.min(), y_h.max(), key=abs) if (Nh !=0 & if_normalize_data) else 1
h_data = jnp.concatenate([x_h, y_h], axis=1)
h_test = jnp.concatenate(
    [x_ref[idx_test, :], h_ref[idx_test][:, np.newaxis]], axis=1)
dataset.update({'h_data': h_data})

if train_mode == 'inverse' or 'forward':

    x_nor = lhs(2, 200000)[:Nres_dar, :]
    x_r = lbt + (ubt - lbt) * x_nor
    y_r = np.zeros((Nres_dar, 1))
    darcy_res = jnp.concatenate([x_r, y_r], axis=1)
    dataset.update({'darcy_res': darcy_res})

    # # # Neumann BC at left
    # x2_bl = np.linspace(lbt[1],ubt[1],Nbl)[:,np.newaxis]
    # x1_bl = lbt[0]*np.ones_like(x2_bl)
    # y_bl = bc_left_dar*np.ones_like(x2_bl)
    # darcy_bl = jnp.concatenate([x1_bl,x2_bl,y_bl],axis=1)
    # dataset.update({'darcy_bl': darcy_bl})

    # Dirichlet BC at left
    x2_bl = x2[:, np.newaxis]
    x1_bl = lbt[0]*np.ones_like(x2_bl)
    y_bl = h_ref_rs[:, 0:1]
    darcy_bl = jnp.concatenate([x1_bl, x2_bl, y_bl], axis=1)
    dataset.update({'darcy_bl': darcy_bl})

    # Dirichlet BC at right
    x2_br = x2[:, np.newaxis]
    x1_br = ubt[0]*np.ones_like(x2_br)
    y_br = h_ref_rs[:, -1:]
    darcy_br = jnp.concatenate([x1_br, x2_br, y_br], axis=1)
    dataset.update({'darcy_br': darcy_br})

    # Neumann BC at top
    x1_bt = x1[:, np.newaxis]
    x2_bt = ubt[1]*np.ones_like(x1_bt)
    y_bt = bc_top_dar*np.ones_like(x1_bt)
    darcy_bt = jnp.concatenate([x1_bt, x2_bt, y_bt], axis=1)
    dataset.update({'darcy_bt': darcy_bt})

    # Neumann BC at below
    x1_bb = x1[:, np.newaxis]
    x2_bb = lbt[1]*np.ones_like(x1_bb)
    y_bb = bc_bot_dar*np.ones_like(x1_bb)
    darcy_bb = jnp.concatenate([x1_bb, x2_bb, y_bb], axis=1)
    dataset.update({'darcy_bb': darcy_bb})

# =============================================================================
# PINN model training
# =============================================================================

model = PinnDarcy2D(layers_y, layers_h, params, dataset,
                    key, norm_y=norm_y, norm_h=norm_h)

if if_load_weights:
    with open(os.path.join(path_f, 'ynet_weights.pkl'), 'rb') as f:
        optim_params_y = pickle.load(f)
    print('====== y DNN loaded =======')
    with open(os.path.join(path_f, 'hnet_weights.pkl'), 'rb') as f:
        optim_params_h = pickle.load(f)
    print('====== h DNN loaded =======')

elif train_mode == 'inverse':

    print('====== Inverse PINN Darcy training starts =======')
    optim_params_y, optim_params_h, opt_state = model.train_darcy_inverse(num_epoch_dar,
                                                                          num_print,
                                                                          if_full_batch=True,
                                                                          batch_size=batch_size,
                                                                          if_lbfgs=if_lbfgs,
                                                                          max_iter_lbfgs=lbfgs_max,
                                                                          rtol=lbfgs_rtol
                                                                          )

    with open(os.path.join(path_f, 'ynet_weights_inverse.pkl'), 'wb') as f:
        pickle.dump(optim_params_y, f)
    with open(os.path.join(path_f, 'hnet_weights_inverse.pkl'), 'wb') as f:
        pickle.dump(optim_params_h, f)

elif train_mode == 'forward':

    print('====== y DNN training starts =======')
    optim_params_y, opt_state_y = model.train_y(num_epoch_y,
                                                num_print,
                                                y_test,
                                                if_full_batch=True,
                                                batch_size=batch_size,
                                                if_lbfgs=if_lbfgs,
                                                max_iter_lbfgs=lbfgs_max,
                                                rtol=lbfgs_rtol
                                                )
    with open(os.path.join(path_f, 'ynet_weights_forward.pkl'), 'wb') as f:
        pickle.dump(optim_params_y, f)
    print('====== Forward PINN Darcy training starts =======')
    optim_params_h, opt_state_h = model.train_darcy_forward(num_epoch_dar,
                                                            num_print,
                                                            if_full_batch=True,
                                                            batch_size=batch_size,
                                                            if_lbfgs=if_lbfgs,
                                                            max_iter_lbfgs=lbfgs_max,
                                                            rtol=lbfgs_rtol
                                                            )
    with open(os.path.join(path_f, 'hnet_weights_forward.pkl'), 'wb') as f:
        pickle.dump(optim_params_h, f)
else:

    print('====== y DNN training starts =======')
    optim_params_y, opt_state_y = model.train_y(num_epoch_y,
                                                num_print,
                                                y_test,
                                                if_full_batch=True,
                                                batch_size=batch_size,
                                                if_lbfgs=if_lbfgs,
                                                max_iter_lbfgs=lbfgs_max,
                                                rtol=lbfgs_rtol
                                                )
    with open(os.path.join(path_f, 'ynet_weights_forward.pkl'), 'wb') as f:
        pickle.dump(optim_params_y, f)
    print('====== h DNN training starts =======')
    model.params_y = optim_params_y
    optim_params_h, opt_state_h = model.train_h(num_epoch_h,
                                                num_print,
                                                h_test,
                                                if_full_batch=True,
                                                batch_size=batch_size,
                                                if_lbfgs=if_lbfgs,
                                                max_iter_lbfgs=lbfgs_max,
                                                rtol=lbfgs_rtol
                                                )
    with open(os.path.join(path_f, 'hnet_weights_forward.pkl'), 'wb') as f:
        pickle.dump(optim_params_h, f)

y_pred = model.predict_y(optim_params_y, x_ref).ravel()
h_pred = model.predict_h(optim_params_h, x_ref).ravel()

print('RL2 Error y: {0: 3e} | Inf Error y: {1: 3e}'.
      format(utils.rl2e(y_pred, y_ref), utils.infe(y_pred, y_ref)))
print('RL2 Error y: {0: 3e} | Inf Error y: {1: 3e}'.
      format(utils.rl2e(y_pred, y_ref), utils.infe(y_pred, y_ref)), file=f_rec)
print('RL2 Error h: {0: 3e} | Inf Error h: {1: 3e}'.
      format(utils.rl2e(h_pred, h_ref), utils.infe(h_pred, h_ref)))
print('RL2 Error h: {0: 3e} | Inf Error h: {1: 3e}'.
      format(utils.rl2e(h_pred, h_ref), utils.infe(h_pred, h_ref)), file=f_rec)

# =============================================================================
# Some visulizations
# =============================================================================

utils.plot_comparison_2d(XX1, XX2, y_pred.reshape(Nx2, Nx1), y_ref.reshape(Nx2, Nx1),
                         points=None, title='$y(\mathbf{x})$', cmap='turbo', dpi=300)
utils.plot_comparison_2d(XX1, XX2, h_pred.reshape(Nx2, Nx1), h_ref.reshape(Nx2, Nx1),
                         points=None, title='$h(\mathbf{x})$', cmap='turbo', dpi=300)

# plt.figure(figsize=(4,4), dpi = 100)
# plt.title('Weight vs time')
# plt.plot(np.arange(0, num_epoch_dar, 100), model.adaptive_weight_bl_log, label = 'left BC')
# plt.plot(np.arange(0, num_epoch_dar, 100), model.adaptive_weight_br_log, label = 'Right BC')
# plt.plot(np.arange(0, num_epoch_dar, 100), model.adaptive_weight_bt_log, label = 'Top BC')
# plt.plot(np.arange(0, num_epoch_dar, 100), model.adaptive_weight_bb_log, label = 'Bottom BC')
# plt.plot(np.arange(0, num_epoch_dar, 100), model.adaptive_weight_y_log, label = 'y')
# plt.plot(np.arange(0, num_epoch_dar, 100), model.adaptive_weight_h_log, label = 'h')
# plt.xlabel('Number of epochs')
# plt.ylabel('Weight')
# plt.yscale('log')
# plt.legend(loc='best')
# plt.show()

f_rec.close()