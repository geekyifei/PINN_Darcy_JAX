#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 15:11:52 2024

@author: yifei_linux
"""
import jax 
import jax.numpy as jnp
from jax import random, grad, vmap, jit, lax
from jax.flatten_util import ravel_pytree
import optax
import optax.tree_utils as otu
from flax import linen as nn
import time
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from tqdm import trange
import jax_dataloader as jdl
from typing import Any, Sequence, Callable, NamedTuple, Optional, Tuple    

# class FNN(nn.Module):
#     layer_sizes: list
    
#     @nn.compact
#     def __call__(self, x):
#         for size in self.layer_sizes[:-1]:
#             x = nn.Dense(size)(x)
#             x = nn.tanh(x)
#         return nn.Dense(self.layer_sizes[-1])(x)

# class FFNN(nn.Module):
#     layer_sizes: list
#     B: jax.Array
    
#     @nn.compact
#     def __call__(self, x):
#         x_proj = jnp.einsum('ij, j->i', self.B, 2.*np.pi*x)
#         z = jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)
#         for size in self.layer_sizes[:-1]:
#             z = nn.Dense(size)(z)
#             z = nn.tanh(z)
#         return nn.Dense(self.layer_sizes[-1])(z)

#Define FNN  
def FNN(layers, activation=jnp.tanh):
    
    # [(W,b),
    #   ...,
    #  (W,b)]

    def init(prng_key): #return a list of (W,b) tuples
    
      def init_layer(key, d_in, d_out):
          key1, key2 = random.split(key)
          glorot_stddev = 1.0 / jnp.sqrt((d_in + d_out) / 2.)
          W = glorot_stddev * random.normal(key1, (d_in, d_out))
          b = jnp.zeros(d_out)
          return W, b
      
      key, *keys = random.split(prng_key, len(layers))
      params = list(map(init_layer, keys, layers[:-1], layers[1:]))
      return params

    def forward(params, inputs):
        Z = inputs
        for W, b in params[:-1]:
            outputs = jnp.dot(Z, W) + b
            Z = activation(outputs)
        W, b = params[-1]
        outputs = jnp.dot(Z, W) + b
        return outputs
    
    return init, forward

def FFNN(layers, B, activation=jnp.tanh):
    
    def init(prng_key):
        
        def init_layer(key, d_in, d_out):
            key1, key2 = random.split(key)
            glorot_stddev = 1.0 / jnp.sqrt((d_in + d_out) / 2.)
            W = glorot_stddev * random.normal(key1, (d_in, d_out))
            b = jnp.zeros(d_out)
            return W, b
        
        key, *keys = random.split(prng_key, len(layers))
        params = list(map(init_layer, keys, layers[:-1], layers[1:]))
        return params
  
    def forward(params, inputs):
        proj = jnp.einsum('j,ij->i', 2* jnp.pi * inputs, B)
        Z = jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1)
        for W, b in params[:-1]:
            outputs = jnp.dot(Z, W) + b
            Z = activation(outputs)
        W, b = params[-1]
        outputs = jnp.dot(Z, W) + b
        return outputs

    return init, forward

class PinnDarcy2D:
    
    def __init__(self, layers_y, layers_h, params, dataset, key, **kwargs):
        
        # General
        self.lbt = params['lbt']  
        self.ubt = params['ubt']  
        self.scale_coe = 0.5
        self.scale = 2 * self.scale_coe / (self.ubt - self.lbt)
        self.norm_h = kwargs.get('norm_h', 1)
        self.norm_y = kwargs.get('norm_y', 1)
        
        # Weights in loss function
        self.lambda_r = params['lambda_res_dar']
        self.lambda_bl = params['lambda_bl']
        self.lambda_br = params['lambda_br']
        self.lambda_bt = params['lambda_bt']
        self.lambda_bb = params['lambda_bb']
        self.lambda_y = params['lambda_y']
        self.lambda_h = params['lambda_h']
        self.lambda_p = params['lambda_p']

        ## Training dataset
        # self.x_y, self.y_y = self.normalize(dataset['y_data'][:, 0:2]), dataset['y_data'][:, 2:3]
        # self.x_h, self.y_h = self.normalize(dataset['h_data'][:, 0:2]), dataset['h_data'][:, 2:3]
        # if params['train_mode'] == 'inverse' or 'forward':
        #     self.x_r, self.y_r = self.normalize(dataset['darcy_res'][:, 0:2]), dataset['darcy_res'][:, 2:3]
        #     self.x_bl, self.y_bl = self.normalize(dataset['darcy_bl'][:, 0:2]), dataset['darcy_bl'][:, 2:3]
        #     self.x_br, self.y_br = self.normalize(dataset['darcy_br'][:, 0:2]), dataset['darcy_br'][:, 2:3]
        #     self.x_bt, self.y_bt = self.normalize(dataset['darcy_bt'][:, 0:2]), dataset['darcy_bt'][:, 2:3]
        #     self.x_bb, self.y_bb = self.normalize(dataset['darcy_bb'][:, 0:2]), dataset['darcy_bb'][:, 2:3]
            
        self.x_y, self.y_y = dataset['y_data'][:, 0:2], dataset['y_data'][:, 2:3]
        self.x_h, self.y_h = dataset['h_data'][:, 0:2], dataset['h_data'][:, 2:3]
        if params['train_mode'] == 'inverse' or 'forward':
            self.x_r, self.y_r = dataset['darcy_res'][:, 0:2], dataset['darcy_res'][:, 2:3]
            self.x_bl, self.y_bl = dataset['darcy_bl'][:, 0:2], dataset['darcy_bl'][:, 2:3]
            self.x_br, self.y_br = dataset['darcy_br'][:, 0:2], dataset['darcy_br'][:, 2:3]
            self.x_bt, self.y_bt = dataset['darcy_bt'][:, 0:2], dataset['darcy_bt'][:, 2:3]
            self.x_bb, self.y_bb = dataset['darcy_bb'][:, 0:2], dataset['darcy_bb'][:, 2:3]

        ## FLAX implementation
        # self.NN_y = FNN(layers_y)
        # self.params_y = self.NN_y.init(keys[0], jnp.ones((1, 2)))  
        # self.NN_h = FNN(layers_h)
        # self.params_h = self.NN_h.init(keys[1], jnp.ones((1, 2)))
        
        # Define DNNs 
        self.key, *keys = random.split(key, num = 4)
        if params['if_fourier_feature']:
            size_ff = 60
            scale_ff = 1.
            self.B = jax.random.normal(keys[2], (size_ff, 2)) * scale_ff
            self.init_k, self.NN_y = FFNN(layers_y, self.B, activation=jnp.tanh)
            self.params_y = self.init_k(keys[0])
            self.init_h, self.NN_h = FNN(layers_h, activation=jnp.tanh)
            self.params_h = self.init_h(keys[1])
        else:
            self.init_k, self.NN_y = FNN(layers_y, activation=jnp.tanh)
            self.params_y = self.init_k(keys[0])
            self.init_h, self.NN_h = FNN(layers_h, activation=jnp.tanh)
            self.params_h = self.init_h(keys[1])
        self.l2_reg = jit(lambda params: jnp.sum((ravel_pytree(params)[0])**2))
        
    def normalize(self, X):
        #normalize input to range[-scale_coe, scale_coe]
        if X.shape[1] == 2:
            return 2.0 * self.scale_coe * (X - self.lbt[0:2])/(self.ubt[0:2] - self.lbt[0:2]) - self.scale_coe
        if X.shape[1] == 3:
            return 2.0 * self.scale_coe * (X - self.lbt)/(self.ubt - self.lbt) - self.scale_coe
    
    @partial(jit, static_argnums=(0,))
    def h_net(self, params_h, x1, x2): 
        inputs = jnp.hstack([x1, x2])
        #outputs = self.NN_h.apply(params_h, inputs)
        outputs = self.NN_h(params_h, inputs)
        return outputs[0]*self.norm_h 
          
    @partial(jit, static_argnums=(0,))
    def y_net(self, params_y, x1, x2): 
        inputs = jnp.hstack([x1, x2])
        #outputs = self.NN_y.apply(params_y, inputs)
        outputs = self.NN_y(params_y, inputs)
        return outputs[0]*self.norm_y
    
    @partial(jit, static_argnums=(0,))
    def qx_net(self, params_y, params_h, x1, x2):
        k = jnp.exp(self.y_net(params_y, x1, x2))
        dhdx = grad(self.h_net, argnums=1)(params_h, x1, x2)
        return -k*dhdx
        
    @partial(jit, static_argnums=(0,))
    def qy_net(self, params_y, params_h, x1, x2):
        k  = jnp.exp(self.y_net(params_y, x1, x2))
        dhdy = grad(self.h_net, argnums=2)(params_h, x1, x2)
        return -k*dhdy
        
    @partial(jit, static_argnums=(0,))
    def darcy_net(self, params_y, params_h, x1, x2):
        d2hdx2 = grad(self.qx_net, argnums=2)(params_y, params_h, x1, x2)
        d2hdy2 = grad(self.qy_net, argnums=3)(params_y, params_h, x1, x2)
        return d2hdx2 + d2hdy2
  
    @partial(jit, static_argnums=(0,))
    def loss_y(self, params_y, x_y, y_y): 
        y_pred = vmap(self.y_net, (None, 0, 0))(params_y, x_y[:,0], x_y[:,1])
        loss_y = jnp.mean((y_pred.flatten() - y_y.flatten())**2)
        return loss_y
    
    @partial(jit, static_argnums=(0,))
    def loss_h(self, params_h, x_h, y_h): 
        h_pred = vmap(self.h_net, (None, 0, 0))(params_h, x_h[:,0], x_h[:,1])
        loss_h = jnp.mean((h_pred.flatten() - y_h.flatten())**2)
        return loss_h
    
    @partial(jit, static_argnums=(0,))
    def loss_r(self, params_y, params_h, x_r, y_r): 
        r_pred = vmap(self.darcy_net, (None, None, 0, 0))(
                        params_y, params_h, x_r[:,0], x_r[:,1]
                        )
        loss_res = jnp.mean((r_pred.flatten() - y_r.flatten())**2)
        return loss_res  
    
    @partial(jit, static_argnums=(0,))
    def loss_bc_dar(self, params_y, params_h, 
                    boundary_idx, bc_type_idx, 
                    x_bc_dar, y_bc_dar
                    ):
        
        def dirichlet_bc(carry):
            params_y, params_h, x_bc_dar, boundary_idx = carry
            bc_pred = vmap(self.h_net, (None, 0, 0))(
                params_h, 
                x_bc_dar[:, 0], x_bc_dar[:, 1]
            )
            return bc_pred
    
        def neumann_bc(carry):
            
            params_y, params_h, x_bc_dar, boundary_idx = carry
            
            def _top(x_bc):
                return vmap(self.qy_net, (None, None, 0, 0))(
                    params_y, params_h, x_bc[:, 0], x_bc[:, 1]
                )
    
            def _bottom(x_bc):
                return vmap(self.qy_net, (None, None, 0, 0))(
                    params_y, params_h, x_bc[:, 0], x_bc[:, 1]
                )
    
            def _left(x_bc):
                return vmap(self.qx_net, (None, None, 0, 0))(
                    params_y, params_h, x_bc[:, 0], x_bc[:, 1]
                )
    
            def _right(x_bc):
                return vmap(self.qx_net, (None, None, 0, 0))(
                    params_y, params_h, x_bc[:, 0], x_bc[:, 1]
                )
    
            bc_pred = lax.switch(
                boundary_idx,
                [_left, _right, _top, _bottom],
                x_bc_dar
            )
            return bc_pred
        
        # type_dict = {'Dirichlet': 0, 'Neumann': 1, 'Robin': 2}
        # boundary_dict = {'left': 0, 'right': 1, 'top': 2, 'bottom': 3}
        
        carry = (params_y, params_h, x_bc_dar, boundary_idx)
        bc_pred = lax.cond(
            bc_type_idx == 0,
            dirichlet_bc,
            neumann_bc,
            operand = carry
        )
    
        loss_bc = jnp.mean((bc_pred.flatten() - y_bc_dar.flatten()) ** 2)
        return loss_bc
    
    @partial(jit, static_argnums=(0,))  
    def loss_darcy_inverse(self, params):
        
        # params_y = params['params_y']
        # params_h = params['params_h']
        params_y = params[0]
        params_h = params[1]
        loss_y = self.loss_y(params_y, self.x_y, self.y_y)
        loss_h = self.loss_h(params_h, self.x_h, self.y_h)
        loss_r = self.loss_r(params_y, params_h, self.x_r, self.y_r)
        loss_bl = self.loss_bc_dar(params_y, params_h, 0, 0, 
                                    self.x_bl, self.y_bl)
        loss_br = self.loss_bc_dar(params_y, params_h, 1, 0, 
                                    self.x_br, self.y_br)
        loss_bt = self.loss_bc_dar(params_y, params_h, 2, 1, 
                                    self.x_bt, self.y_bt)
        loss_bb = self.loss_bc_dar(params_y, params_h, 3, 1, 
                                    self.x_bb, self.y_bb)
        loss = self.lambda_y*loss_y + \
               self.lambda_h*loss_h + \
               self.lambda_r*loss_r + \
               self.lambda_bl*loss_bl + \
               self.lambda_br*loss_br + \
               self.lambda_bt*loss_bt + \
               self.lambda_bb*loss_bb + \
               self.lambda_p*(self.l2_reg(params_y) + self.l2_reg(params_h))
               
        return loss 
    
    @partial(jit, static_argnums=(0,))  
    def loss_darcy_forward(self, params_h):
        
        loss_r = self.loss_r(self.params_y, params_h, self.x_r, self.y_r)
        loss_bl = self.loss_bc_dar(self.params_y, params_h, 0, 0, 
                                    self.x_bl, self.y_bl)
        loss_br = self.loss_bc_dar(self.params_y, params_h, 1, 0, 
                                    self.x_br, self.y_br)
        loss_bt = self.loss_bc_dar(self.params_y, params_h, 2, 1, 
                                    self.x_bt, self.y_bt)
        loss_bb = self.loss_bc_dar(self.params_y, params_h, 3, 1, 
                                    self.x_bb, self.y_bb)
        loss = self.lambda_r*loss_r + \
               self.lambda_bl*loss_bl + \
               self.lambda_br*loss_br + \
               self.lambda_bt*loss_bt + \
               self.lambda_bb*loss_bb 
               
        return loss
    
    def train_y(self, num_epoch, num_print, y_test,
                if_full_batch, batch_size, if_lbfgs, max_iter_lbfgs, rtol):
        
        '''
        Train y DNN with measurements in the data-driven manner

        Args:
          num_epoch: number of training epochs
          num_print: intervals for printing options.
          y_test: testing data for y, can be None
          if_full_batch: if to use full-batch of data for training
          if_lbfgs: if to use L-BFGS after Adam
          max_iter: max iterations of L-BFGS training
          rtol: min tolerance for gradient of L-BFGS loss
        
        Returns:
          params_y: optimal parameters of the model.
        '''
        
        @partial(jit, static_argnums=())
        def step(params_y, opt_state_y):

            grads_y = grad(self.loss_y, argnums=0)(params_y, self.x_y, self.y_y)
            updates_y, opt_state_y = optimizer.update(grads_y, opt_state_y)
            params_y = optax.apply_updates(params_y, updates_y)
            
            return params_y, opt_state_y
        
        lr_schedule = optax.schedules.exponential_decay(1e-3, transition_steps=1000, decay_rate=0.9)
        optimizer = optax.adam(lr_schedule) # or adamw equipped with weight decay
        opt_state_y = optimizer.init(self.params_y)
        
        loss_train_rec = jnp.zeros((num_epoch + max_iter_lbfgs))
        loss_test_rec = jnp.zeros((num_epoch + max_iter_lbfgs))
        pbar = trange(num_epoch)
        
        since = time.time()
        if if_full_batch == True: # Full-batch
            for it in pbar:
                self.params_y, opt_state_y = step(self.params_y, opt_state_y)
                loss_value = self.loss_y(self.params_y, self.x_y, self.y_y)
                loss_train_rec = loss_train_rec.at[it].set(loss_value)
                if y_test is not None:
                    test_loss = self.loss_y(self.params_y, 
                                            self.normalize(y_test[:,0:2]), 
                                            y_test[:,2:3])
                    loss_test_rec = loss_test_rec.at[it].set(test_loss)
                if it % num_print == 0:
                    pbar.set_postfix({'Loss': loss_value})
        
        else: # Mini-batch
        
            @jax.jit
            def step_batch(params, opt_state, x_batch, y_batch):
                
                grads = grad(self.loss_y, argnums=0)(params, x_batch, y_batch)
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)

                return params, opt_state
        
            num_batch_y = self.y_y.shape[0] // batch_size
            dataloader = jdl.DataLoader(
                jdl.ArrayDataset(self.x_y, self.y_y),
                backend='jax', 
                batch_size=batch_size, 
                shuffle=True, 
                drop_last=True, 
            )

            for it in pbar:
                epoch_loss = 0.
                for batch in dataloader:
                    x_batch, y_batch = batch
                    self.params_y, opt_state_y = step_batch(self.params_y, opt_state_y, x_batch, y_batch)
                    batch_loss = self.loss_y(self.params_y, x_batch, y_batch)
                    epoch_loss += batch_loss
                epoch_loss /= num_batch_y 
                if y_test is not None:
                    test_loss = self.loss_y(self.params_y, 
                                            self.normalize(y_test[:,0:2]), 
                                            y_test[:,2:3])
                    loss_test_rec = loss_test_rec.at[it].set(test_loss)
                loss_train_rec = loss_train_rec.at[it].set(epoch_loss)

                if it % num_print == 0:
                    pbar.set_postfix({'loss': f"{epoch_loss:.4e}" })
        
        time_elapsed = time.time() - since
        print(f'Adam time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        
        if if_lbfgs == True:
            
            since = time.time()
            lbfgs_opt = optax.lbfgs()
            opt_state_y = lbfgs_opt.init(self.params_y)
            loss_fn_y = lambda params_y: self.loss_y(params_y, self.x_y, self.y_y)
            value_and_grad_fn = optax.value_and_grad_from_state(loss_fn_y)
            
            @jax.jit
            def step_lbfgs(carry):
                
                params_y, opt_state_y, loss_train_rec = carry
                value, grads = value_and_grad_fn(params_y, state=opt_state_y)
                updates_y, opt_state_y = lbfgs_opt.update(grads, opt_state_y, params_y, 
                                                          value=value, grad=grad, value_fn=self.loss_y
                                                          )
                params_y = optax.apply_updates(params_y, updates_y)
                iter_num = otu.tree_get(opt_state_y, 'count')
                loss_train_rec = loss_train_rec.at[num_epoch + iter_num].set(value)
                
                return (params_y, opt_state_y)
            
            @jax.jit
            def stopping_criterion(carry):
                params_y, opt_state_y = carry
                iter_num = otu.tree_get(opt_state_y, 'count')
                grad_y = otu.tree_get(opt_state_y, 'grad')
                grad_l2_norm = otu.tree_l2_norm(grad_y)
                return (iter_num == 0) | ((iter_num < max_iter_lbfgs) & (grad_l2_norm >= rtol))
          
            init_carry = (self.params_y, opt_state_y, loss_train_rec)
            self.params_y, opt_state_y, loss_train_rec = jax.lax.while_loop(
                                            stopping_criterion, step_lbfgs, init_carry
                                            )
                    
            time_elapsed = time.time() - since
            print(f'L-BFGS time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        
            lbfgs_iter_num = otu.tree_get(opt_state_y, 'count')
            total_epoch = num_epoch + lbfgs_iter_num
            plt.figure(figsize=(4,4), dpi = 100)
            plt.title('Training Loss')
            plt.plot(jnp.arange(total_epoch),loss_train_rec[:total_epoch], label = 'Loss') 
            plt.xlabel('Number of epochs')
            plt.ylabel('Loss')
            plt.yscale('log')
            plt.legend(loc='best')
            plt.show()
        
        else:
            
            total_epoch = num_epoch
            plt.figure(figsize=(4,4), dpi = 100)
            plt.plot(jnp.arange(total_epoch),loss_train_rec[:total_epoch], label = 'Train Loss') 
            if y_test is not None:
                plt.plot(jnp.arange(total_epoch),loss_test_rec[:total_epoch], label = 'Test Loss')
            plt.xlabel('Number of epochs')
            plt.ylabel('Loss')
            plt.yscale('log')
            plt.legend(loc='best')
            plt.show()
        
        return self.params_y, opt_state_y
    
    def train_h(self, num_epoch, num_print, h_test,
                if_full_batch, batch_size, if_lbfgs, max_iter_lbfgs, rtol):
        
        '''
        Train h DNN with measurements in the data-driven manner

        Args:
          num_epoch: number of training epochs
          num_print: intervals for printing options.
          h_test: testing data for h, can be None
          if_full_batch: if to use full-batch of data for training
          batch_size: mini-batch size
          if_lbfgs: if to use L-BFGS after Adam
          max_iter_lbfgs: max iterations of L-BFGS training
          rtol: min tolerance for gradient of L-BFGS loss
        
        Returns:
          params_h: optimal parameters of the h DNN model.
        '''
        
        @partial(jit, static_argnums=())
        def step(params_h, opt_state_h):
            grads_h = grad(self.loss_h, argnums=0)(params_h, self.x_h, self.y_h)
            updates_h, opt_state_h = optimizer.update(grads_h, opt_state_h)
            params_h = optax.apply_updates(params_h, updates_h)
            return params_h, opt_state_h
        
        lr_schedule = optax.schedules.exponential_decay(1e-3, transition_steps=1000, decay_rate=0.9)
        optimizer = optax.adam(lr_schedule) # or adamw
        opt_state_h = optimizer.init(self.params_h)
        
        since = time.time()
        loss_train_rec = jnp.zeros((num_epoch + max_iter_lbfgs))
        loss_test_rec = jnp.zeros((num_epoch + max_iter_lbfgs))
        pbar = trange(num_epoch)
        
        # Main training loop
        if if_full_batch == True:
            for it in pbar:
                self.params_h, opt_state_h = step(self.params_h, opt_state_h)
                loss_value = self.loss_h(self.params_h, self.x_h, self.y_h)
                loss_train_rec.append(loss_value)
                if h_test is not None:
                    test_loss = self.loss_h_batch(self.params_h, 
                                                  self.normalize(h_test[:,0:2]), 
                                                  h_test[:,2:3])
                    loss_test_rec.append(test_loss)
                if it % num_print == 0:
                    pbar.set_postfix({'Loss': loss_value})
                    
        else: # Mini-batch
        
            @jax.jit
            def step_batch(params, opt_state, x_batch, y_batch):
                
                grads = grad(self.loss_h, argnums=0)(params, x_batch, y_batch)
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)

                return params, opt_state
        
            num_batch_h = self.y_h.shape[0] // batch_size
            dataloader = jdl.DataLoader(
                jdl.ArrayDataset(self.x_h, self.y_h),
                backend='jax', 
                batch_size=batch_size, 
                shuffle=True, 
                drop_last=True, 
            )

            for it in pbar:
                epoch_loss = 0.
                for batch in dataloader:
                    x_batch, y_batch = batch
                    self.params_h, opt_state_h = step_batch(self.params_h, opt_state_h, x_batch, y_batch)
                    batch_loss = self.loss_h(self.params_h, x_batch, y_batch)
                    epoch_loss += batch_loss
                epoch_loss /= num_batch_h 
                if h_test is not None:
                    test_loss = self.loss_h(self.params_h, 
                                            self.normalize(h_test[:,0:2]), 
                                            h_test[:,2:3])
                    loss_test_rec = loss_test_rec.at[it].set(test_loss)
                loss_train_rec = loss_train_rec.at[it].set(epoch_loss)

        time_elapsed = time.time() - since
        print(f'Training time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        
        if if_lbfgs == True:
            
            since = time.time()
            lbfgs_opt = optax.lbfgs()
            opt_state_h = lbfgs_opt.init(self.params_h)
            loss_fn_h = lambda params_h: self.loss_h(params_h, self.x_h, self.y_h)
            value_and_grad_fn = optax.value_and_grad_from_state(loss_fn_h)
            #value_and_grad_fn = optax.value_and_grad_from_state(self.loss_h)
            
            @jax.jit
            def step_lbfgs(carry):
                
                params_h, opt_state_h, loss_train_rec = carry
                value, grads = value_and_grad_fn(params_h, state=opt_state_h)
                updates_h, opt_state_h = lbfgs_opt.update(grads, opt_state_h, params_h, 
                                                          value=value, grad=grads, value_fn=self.loss_h
                                                          )
                params_h = optax.apply_updates(params_h, updates_h)
                iter_num = otu.tree_get(opt_state_h, 'count')
                loss_train_rec = loss_train_rec.at[num_epoch + iter_num].set(value)
                
                return (params_h, opt_state_h, loss_train_rec)
            
            @jax.jit
            def stopping_criterion(carry):
                params_h, opt_state_h = carry
                iter_num = otu.tree_get(opt_state_h, 'count')
                grad_h = otu.tree_get(opt_state_h, 'grad')
                grad_l2_norm = otu.tree_l2_norm(grad_h)
                return (iter_num == 0) | ((iter_num < max_iter_lbfgs) & (grad_l2_norm >= rtol))
          
            init_carry = (self.params_h, opt_state_h, loss_train_rec)
            self.params_h, opt_state_h, loss_train_rec = jax.lax.while_loop(
                                            stopping_criterion, step_lbfgs, init_carry
                                            )
                    
            time_elapsed = time.time() - since
            print(f'L-BFGS time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        
            lbfgs_iter_num = otu.tree_get(opt_state_h, 'count')
            total_epoch = num_epoch + lbfgs_iter_num
            plt.figure(figsize=(4,4), dpi = 100)
            plt.title('Training Loss')
            plt.plot(jnp.arange(total_epoch),loss_train_rec[:total_epoch], label = 'Loss') 
            plt.xlabel('Number of epochs')
            plt.ylabel('Loss')
            plt.yscale('log')
            plt.legend(loc='best')
            plt.show()
        
        else:
            
            total_epoch = num_epoch
            plt.figure(figsize=(4,4), dpi = 100)
            plt.plot(jnp.arange(total_epoch),loss_train_rec[:total_epoch], label = 'Train Loss') 
            if h_test is not None:
                plt.plot(jnp.arange(total_epoch),loss_test_rec[:total_epoch], label = 'Test Loss')
            plt.xlabel('Number of epochs')
            plt.ylabel('Loss')
            plt.yscale('log')
            plt.legend(loc='best')
            plt.show()
        
        return self.params_h, opt_state_h
    
    def train_darcy_forward(self, num_epoch, num_print, if_full_batch, batch_size,
                            if_lbfgs, max_iter_lbfgs, rtol
                            ):
        
        '''
        Train y DNN with measurements in the data-driven manner, 
            and then train the h DNN using PINN-Darcy

        Args:
          num_epoch: number of training epochs
          num_print: intervals for printing options.
          h_test: testing data for h, can be None
          if_full_batch: if to use full-batch of data for training
          batch_size: mini-batch size
          if_lbfgs: if to use L-BFGS after Adam
          max_iter_lbfgs: max iterations of L-BFGS training
          rtol: min tolerance for gradient of L-BFGS loss
        
        Returns:
          params_h: optimal parameters of the h DNN model.
        '''
        
        @partial(jit, static_argnums=())
        def step(params_h, opt_state_h):  
            grads_h = grad(self.loss_darcy_forward, argnums=0)(params_h)
            updates_h, opt_state_h = optimizer.update(grads_h, opt_state_h)
            params_h = optax.apply_updates(params_h, updates_h)
            return params_h, opt_state_h
        
        since = time.time()
        loss_rec = jnp.zeros((num_epoch + max_iter_lbfgs))
        loss_r_rec = jnp.zeros((num_epoch + max_iter_lbfgs))
        loss_bc_rec = jnp.zeros((num_epoch + max_iter_lbfgs))
        pbar = trange(num_epoch)
        lr_schedule = optax.schedules.exponential_decay(1e-3, transition_steps=1000, decay_rate=0.9)
        optimizer = optax.adam(lr_schedule) # or adamw
        opt_state_h = optimizer.init(self.params_h)
        
        # Main training loop
        if if_full_batch == True:
            for it in pbar:
                self.params_h, opt_state_h = step(self.params_h, opt_state_h)
                loss_value = self.loss_darcy_forward(self.params_h)
                r_loss = self.lambda_r*self.loss_r(self.params_y, self.params_h, self.x_r, self.y_r)
                bc_loss = loss_value - r_loss
                loss_rec = loss_rec.at[it].set(loss_value)
                loss_r_rec = loss_r_rec.at[it].set(r_loss)
                loss_bc_rec = loss_bc_rec.at[it].set(bc_loss)
                if it % num_print == 0:
                    pbar.set_postfix({
                            'loss': f"{loss_value:.4e}", 
                            'PDE loss': f"{r_loss:.4e}",
                            'BC loss': f"{bc_loss:.4e}"
                            })

        time_elapsed = time.time() - since
        print(f'Training time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        
        if if_lbfgs == True:
            
            since = time.time()
            lbfgs_opt = optax.lbfgs()
            opt_state_h = lbfgs_opt.init(self.params_h)
            value_and_grad_fn = optax.value_and_grad_from_state(self.loss_darcy_forward)
            
            @jax.jit
            def step_lbfgs(carry):
                params_h, opt_state_h = carry
                value, grads = value_and_grad_fn(params_h, state=opt_state_h)
                updates_h, opt_state_h = lbfgs_opt.update(grads, opt_state_h, params_h, 
                                                          value=value, grad=grads, value_fn=self.loss_darcy_forward
                                                          )
                params_h = optax.apply_updates(params_h, updates_h)
                return (params_h, opt_state_h)
            
            @jax.jit
            def stopping_criterion(carry):
                params_h, opt_state_h = carry
                iter_num = otu.tree_get(opt_state_h, 'count')
                grad_h = otu.tree_get(opt_state_h, 'grad')
                grad_l2_norm = otu.tree_l2_norm(grad_h)
                return (iter_num == 0) | ((iter_num < max_iter_lbfgs) & (grad_l2_norm >= rtol))
          
            init_carry = (self.params_h, opt_state_h)
            self.params_h, opt_state_h = jax.lax.while_loop(
                                            stopping_criterion, step_lbfgs, init_carry
                                            )
                    
            time_elapsed = time.time() - since
            print(f'L-BFGS time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        
            lbfgs_iter_num = otu.tree_get(opt_state_h, 'count')
            total_epoch = num_epoch + lbfgs_iter_num
            plt.figure(figsize=(4,4), dpi = 100)
            plt.title('Training Loss')
            plt.plot(jnp.arange(total_epoch),loss_rec[:total_epoch], label = 'Total Loss') 
            plt.plot(jnp.arange(total_epoch),loss_r_rec[:total_epoch], label = 'PDE Loss')
            plt.plot(jnp.arange(total_epoch),loss_bc_rec[:total_epoch], label = 'BC Loss')
            plt.xlabel('Number of epochs')
            plt.ylabel('Loss')
            plt.yscale('log')
            plt.legend(loc='best')
            plt.show()
        
        else:
            
            total_epoch = num_epoch
            plt.figure(figsize=(4,4), dpi = 100)
            plt.title('Training Loss')
            plt.plot(jnp.arange(total_epoch),loss_rec[:total_epoch], label = 'Total Loss') 
            plt.plot(jnp.arange(total_epoch),loss_r_rec[:total_epoch], label = 'PDE Loss')
            plt.plot(jnp.arange(total_epoch),loss_bc_rec[:total_epoch], label = 'BC Loss')
            plt.xlabel('Number of epochs')
            plt.ylabel('Loss')
            plt.yscale('log')
            plt.legend(loc='best')
            plt.show()
        
        return self.params_h, opt_state_h
    
    def train_darcy_inverse(self, num_epoch, num_print, if_full_batch, batch_size,
                                if_lbfgs, max_iter_lbfgs, rtol
                                ):
        
        '''
        Train y and h DNN jointly with PINN

        Args:
          num_epoch: number of training epochs
          num_print: intervals for printing options.
          h_test: testing data for h, can be None
          if_full_batch: if to use full-batch of data for training
          batch_size: mini-batch size
          if_lbfgs: if to use L-BFGS after Adam
          max_iter_lbfgs: max iterations of L-BFGS training
          rtol: min tolerance for gradient of L-BFGS loss
        
        Returns:
          params_y, params_h: optimal parameters of the y, h DNN model.
        '''
        
        @jax.jit
        def step(params, opt_state):
            grads = grad(self.loss_darcy_inverse)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state
        
        # params =  dict({'params_y': self.params_y, 
        #                 'params_h': self.params_h})
        params =  [self.params_y, self.params_h]
        
        lr_schedule = optax.schedules.exponential_decay(1e-3, transition_steps=5000, decay_rate=0.9)
        optimizer = optax.adam(lr_schedule) # or adamw
        opt_state = optimizer.init(params)
        
        since = time.time()
        loss_rec = jnp.zeros((num_epoch + max_iter_lbfgs))
        loss_r_rec = jnp.zeros((num_epoch + max_iter_lbfgs))
        loss_bc_rec = jnp.zeros((num_epoch + max_iter_lbfgs))
        pbar = trange(num_epoch)
        
        if if_full_batch == True:
            
            for it in pbar:
                
                params, opt_state = step(params, opt_state)
                loss_value = self.loss_darcy_inverse(params)
                #r_loss = self.lambda_r*self.loss_r(params['params_y'], params['params_h'], self.x_r, self.y_r)
                r_loss = self.lambda_r*self.loss_r(params[0], params[1], self.x_r, self.y_r)
                bc_loss = loss_value - r_loss
                loss_rec = loss_rec.at[it].set(loss_value)
                loss_r_rec = loss_r_rec.at[it].set(r_loss)
                loss_bc_rec = loss_bc_rec.at[it].set(bc_loss)
                
                if it % num_print == 0:

                    pbar.set_postfix({
                            'loss': f"{loss_value:.4e}", 
                            'PDE loss': f"{r_loss:.4e}",
                            'BC loss': f"{bc_loss:.4e}"
                            })
        
            time_elapsed = time.time() - since
            print(f'Adam Training time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        
        else:
            
            @jax.jit
            def step_batch(params, opt_state, x_batch, y_batch):
                grads = grad(self.loss_darcy_inverse_batch, argnums=0)(params, x_batch, y_batch)
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
                return params, opt_state
            
            (dataloader_r, dataloader_y, dataloader_h, 
             dataloader_bl, dataloader_br, 
             dataloader_bt, dataloader_bb) = self.get_dataloader(batch_size)

        
        if if_lbfgs == True:
            
            since = time.time()
            lbfgs_opt = optax.lbfgs()
            opt_state = lbfgs_opt.init(params)
            value_and_grad_fn = optax.value_and_grad_from_state(self.loss_darcy_inverse)
            
            @jax.jit
            def step_lbfgs(carry):
                
                params, opt_state, loss_rec, loss_r_rec, loss_bc_rec  = carry
                value, grads = value_and_grad_fn(params, state=opt_state)
                updates, opt_state = lbfgs_opt.update(grads, opt_state, params, 
                                                          value=value, grad=grads, value_fn=self.loss_darcy_inverse
                                                          )
                params = optax.apply_updates(params, updates)
                
                iter_num = otu.tree_get(opt_state, 'count')
                r_loss = self.loss_r(params['params_y'], params['params_h'], self.x_r, self.y_r)
                bc_loss = loss_value - r_loss
                loss_rec = loss_rec.at[num_epoch + iter_num].set(loss_value)
                loss_r_rec = loss_r_rec.at[num_epoch + iter_num].set(r_loss)
                loss_bc_rec = loss_bc_rec.at[num_epoch + iter_num].set(bc_loss)

                jax.lax.cond(
                    iter_num % num_print == 0,
                    lambda _: jax.debug.print("Iteration {i}: Loss = {l}", i=iter_num, l=value),
                    lambda _: None,
                    operand=None,
                )

                return params, opt_state, loss_rec, loss_r_rec, loss_bc_rec 
            
            @jax.jit
            def stopping_criterion(carry):
                params, opt_state, loss_rec, loss_r_rec, loss_bc_rec = carry
                iter_num = otu.tree_get(opt_state, 'count')
                grads = otu.tree_get(opt_state, 'grad')
                grad_l2_norm = otu.tree_l2_norm(grads)
                return (iter_num == 0) | ((iter_num < max_iter_lbfgs) & (grad_l2_norm >= rtol))
          
            init_carry = (params, opt_state, loss_rec, loss_r_rec, loss_bc_rec)
            params, opt_state,  loss_rec, loss_r_rec, loss_bc_rec = jax.lax.while_loop(
                                            stopping_criterion, step_lbfgs, init_carry
                                            )
                    
            time_elapsed = time.time() - since
            print(f'L-BFGS time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        
            lbfgs_iter_num = otu.tree_get(opt_state, 'count')
            total_epoch = num_epoch + lbfgs_iter_num
            plt.figure(figsize=(4,4), dpi = 100)
            plt.title('Training Loss')
            plt.plot(jnp.arange(total_epoch),loss_rec[:total_epoch], label = 'Total Loss') 
            plt.plot(jnp.arange(total_epoch),loss_r_rec[:total_epoch], label = 'PDE Loss')
            plt.plot(jnp.arange(total_epoch),loss_bc_rec[:total_epoch], label = 'BC Loss')
            plt.xlabel('Number of epochs')
            plt.ylabel('Loss')
            plt.yscale('log')
            plt.legend(loc='best')
            plt.show()
        
        else:
            
            total_epoch = num_epoch
            plt.figure(figsize=(4,4), dpi = 100)
            plt.title('Training Loss')
            plt.plot(jnp.arange(total_epoch),loss_rec[:total_epoch], label = 'Total Loss') 
            plt.plot(jnp.arange(total_epoch),loss_r_rec[:total_epoch], label = 'PDE Loss')
            plt.plot(jnp.arange(total_epoch),loss_bc_rec[:total_epoch], label = 'BC Loss')
            plt.xlabel('Number of epochs')
            plt.ylabel('Loss')
            plt.yscale('log')
            plt.legend(loc='best')
            plt.show()
        
        #return params['params_y'], params['params_h'], opt_state
        return params[0], params[1], opt_state
    
    def get_dataloader(self, batch_size):
        
        Ny = self.y_y.shape[0]
        Nh = self.y_h.shape[0]
        Nr = self.y_r_dar.shape[0]
        Nbl = self.y_bl_dar.shape[0]
        Nbt = self.y_bt_dar.shape[0]

        num_batch_r = Nr // batch_size
        batch_size_y = Ny // num_batch_r
        batch_size_h = Nh // num_batch_r
        batch_size_bl = Nbl // num_batch_r
        batch_size_bt = Nbt // num_batch_r
            
        dataloader_r = jdl.DataLoader(
            jdl.ArrayDataset(self.x_r, self.y_r), 
            backend='jax', 
            batch_size=batch_size, 
            shuffle=True,
            drop_last=True, 
        )
        dataloader_y = jdl.DataLoader(
            jdl.ArrayDataset(self.x_y, self.y_y), 
            backend='jax', 
            batch_size=batch_size_y, 
            shuffle=True,
            drop_last=True, 
        )
        dataloader_h = jdl.DataLoader(
            jdl.ArrayDataset(self.x_h, self.y_h), 
            backend='jax', 
            batch_size=batch_size_h, 
            shuffle=True,
            drop_last=True, 
        )
        dataloader_bl = jdl.DataLoader(
            jdl.ArrayDataset(self.x_bl, self.y_bl), 
            backend='jax', 
            batch_size=batch_size_bl, 
            shuffle=True,
            drop_last=True, 
        )
        dataloader_br = jdl.DataLoader(
            jdl.ArrayDataset(self.x_br, self.y_br), 
            backend='jax', 
            batch_size=batch_size_bl, 
            shuffle=True,
            drop_last=True, 
        )
        dataloader_bt = jdl.DataLoader(
            jdl.ArrayDataset(self.x_bt, self.y_bt), 
            backend='jax', 
            batch_size=batch_size_bt, 
            shuffle=True,
            drop_last=True, 
        )
        dataloader_bb = jdl.DataLoader(
            jdl.ArrayDataset(self.x_bb, self.y_bb), 
            backend='jax', 
            batch_size=batch_size_bt, 
            shuffle=True,
            drop_last=True, 
        )
        
        return (dataloader_r, dataloader_y, dataloader_h,
                dataloader_bl, dataloader_br, dataloader_bt, dataloader_bb)
    
    # =========================================================================
    # DNN inference
    # =========================================================================
    
    @partial(jit, static_argnums=(0,))
    def predict_h(self, params_h, x_star):
        #x_star = self.normalize(x_star)
        return vmap(self.h_net, (None, 0, 0))(params_h, x_star[:,0], x_star[:,1])
        
    @partial(jit, static_argnums=(0,))
    def predict_y(self, params_y, x_star):
        #x_star = self.normalize(x_star)
        return vmap(self.y_net, (None, 0, 0))(params_y, x_star[:,0], x_star[:,1])

    @partial(jit, static_argnums=(0,))
    def predict_r(self, params_y, params_h, x_star):
        #x_star = self.normalize(x_star)
        return vmap(self.darcy_net, (None, None, 0, 0))(params_y, params_h, x_star[:,0], x_star[:,1])
        
# =============================================================================
#     Legacy code
# =============================================================================
    
    # @partial(jit, static_argnums=(0,))
    # def loss_bl(self, params_y, params_h):
    #     bl_pred = vmap(self.qx_net, (None, None, 0, 0))(
    #                     params_y, params_h, self.x_bl[:,0], self.x_bl[:,1]
    #                     )
    #     loss_bl = jnp.mean((bl_pred.flatten() - self.y_bl.flatten())**2)
    #     return loss_bl 
    
    # @partial(jit, static_argnums=(0,))
    # def loss_br(self, params_h):
    #     h_pred = vmap(self.h_net, (None, 0, 0))(
    #                     params_h, self.x_br[:,0], self.x_br[:,1]
    #                     )
    #     loss_br = jnp.mean((h_pred.flatten() - self.y_br.flatten())**2)
    #     return loss_br
    
    # @partial(jit, static_argnums=(0,))
    # def loss_bb(self, params_y, params_h):
    #     bb_pred = vmap(self.qy_net, (None, None, 0, 0))(
    #                     params_y, params_h, self.x_bb[:,0], self.x_bb[:,1]
    #                     )
    #     loss_bb = jnp.mean((bb_pred.flatten() - self.y_bb.flatten())**2)
    #     return loss_bb
    
    # @partial(jit, static_argnums=(0,))
    # def loss_bt(self, params_y, params_h):
    #     bt_pred = vmap(self.qy_net, (None, None, 0, 0))(
    #                     params_y, params_h, self.x_bt[:,0], self.x_bt[:,1]
    #                     )
    #     loss_bt = jnp.mean((bt_pred.flatten() - self.y_bt.flatten())**2)
    #     return loss_bt