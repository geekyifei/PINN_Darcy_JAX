#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 19:24:16 2024

@author: yifei_linux
"""

import jax 
import jax.numpy as jnp
from jax import  grad, vmap, jit, tree_util

@jit
def tree_norm(pytree):
    squared_tree = jax.tree_util.tree_map(lambda x: jnp.sum(x**2), pytree)
    total_squared = jax.tree_util.tree_reduce(lambda x, y: x + y, squared_tree, initializer=0.0)
    return jnp.sqrt(total_squared)

def tree_dot(a, b):
  return sum([jnp.sum(e1 * e2) for e1, e2 in
              zip(jax.tree.leaves(a), jax.tree.leaves(b))])

@jit
def tree_add(a, b):
  return jax.tree.map(lambda e1, e2: e1+e2, a, b)

@jit
def tree_diff(a, b):
  return jax.tree.map(lambda p_a, p_b: p_a - p_b, a, b)