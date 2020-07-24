#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Saves features based on volume sensitivity

For usage information, call with --help.

Author: Vinod Subramanian
"""

from __future__ import print_function

import sys
import os
import io
from argparse import ArgumentParser
import json
import pickle 

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
floatX = np.float32
from scipy.special import kl_div

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from progress import progress
from simplecache import cached
import audio
from labels import create_aligned_targets
import model
import augment
import config

from utils import clip_eta, optimize_linear

def fast_gradient_method(model_fn, x, eps, norm,
                         clip_min=None, clip_max=None, y=None, targeted=False, sanity_checks=False):
  """
  PyTorch implementation of the Fast Gradient Method.
  :param model_fn: a callable that takes an input tensor and returns the model logits.
  :param x: input tensor.
  :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
  :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
  :param clip_min: (optional) float. Minimum float value for adversarial example components.
  :param clip_max: (optional) float. Maximum float value for adversarial example components.
  :param y: (optional) Tensor with true labels. If targeted is true, then provide the
            target label. Otherwise, only provide this parameter if you'd like to use true
            labels when crafting adversarial samples. Otherwise, model predictions are used
            as labels to avoid the "label leaking" effect (explained in this paper:
            https://arxiv.org/abs/1611.01236). Default is None.
  :param targeted: (optional) bool. Is the attack targeted or untargeted?
            Untargeted, the default, will try to make the label incorrect.
            Targeted will instead try to move in the direction of being more like y.
  :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
            memory or for unit tests that intentionally pass strange input)
  :return: a tensor for the adversarial example
  """
  if norm not in [np.inf, 1, 2]:
    raise ValueError("Norm order must be either np.inf, 1, or 2, got {} instead.".format(norm))
  if eps < 0:
    raise ValueError("eps must be greater than or equal to 0, got {} instead".format(eps))
  if eps == 0:
    return x
  if clip_min is not None and clip_max is not None:
    if clip_min > clip_max:
      raise ValueError(
          "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
              clip_min, clip_max))

  asserts = []

  # If a data range was specified, check that the input was in that range
  if clip_min is not None:
    assert_ge = torch.all(torch.ge(x, torch.tensor(clip_min, device=x.device, dtype=x.dtype)))
    asserts.append(assert_ge)

  if clip_max is not None:
    assert_le = torch.all(torch.le(x, torch.tensor(clip_max, device=x.device, dtype=x.dtype)))
    asserts.append(assert_le)

  # x needs to be a leaf variable, of floating point type and have requires_grad being True for
  # its grad to be computed and stored properly in a backward call
  x = x.clone().detach().to(torch.float).requires_grad_(True)
  if y is None:
    # Using model predictions as ground truth to avoid label leaking
    _, y = torch.max(model_fn(x), 1)

  # Compute loss
  loss_fn = torch.nn.BCELoss()
  loss = loss_fn(model_fn(x), y)
  # If attack is targeted, minimize loss of target label rather than maximize loss of correct label
  if targeted:
    loss = -loss

  # Define gradient of loss wrt input
  loss.backward()
  optimal_perturbation = optimize_linear(x.grad, eps, norm)

  # Add perturbation to original example to obtain adversarial example
  adv_x = x + optimal_perturbation

  # If clipping is needed, reset all values outside of [clip_min, clip_max]
  if (clip_min is not None) or (clip_max is not None):
    if clip_min is None or clip_max is None:
      raise ValueError(
          "One of clip_min and clip_max is None but we don't currently support one-sided clipping")
    adv_x = torch.clamp(adv_x, clip_min, clip_max)

  if sanity_checks:
    assert np.all(asserts)
  return adv_x

def projected_gradient_descent(model_fn, x, eps, eps_iter, nb_iter, norm,
                               clip_min=None, clip_max=None, y=None, targeted=False,
                               threshold=None,rand_init=True, rand_minmax=None,
                               sanity_checks=True):
  """
  This class implements either the Basic Iterative Method
  (Kurakin et al. 2016) when rand_init is set to False. or the
  Madry et al. (2017) method if rand_init is set to True.
  Paper link (Kurakin et al. 2016): https://arxiv.org/pdf/1607.02533.pdf
  Paper link (Madry et al. 2017): https://arxiv.org/pdf/1706.06083.pdf
  :param model_fn: a callable that takes an input tensor and returns the model logits.
  :param x: input tensor.
  :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
  :param eps_iter: step size for each attack iteration
  :param nb_iter: Number of attack iterations.
  :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
  :param clip_min: (optional) float. Minimum float value for adversarial example components.
  :param clip_max: (optional) float. Maximum float value for adversarial example components.
  :param y: (optional) Tensor with true labels. If targeted is true, then provide the
            target label. Otherwise, only provide this parameter if you'd like to use true
            labels when crafting adversarial samples. Otherwise, model predictions are used
            as labels to avoid the "label leaking" effect (explained in this paper:
            https://arxiv.org/abs/1611.01236). Default is None.
  :param targeted: (optional) bool. Is the attack targeted or untargeted?
            Untargeted, the default, will try to make the label incorrect.
            Targeted will instead try to move in the direction of being more like y.
  :param rand_init: (optional) bool. Whether to start the attack from a randomly perturbed x.
  :param rand_minmax: (optional) bool. Support of the continuous uniform distribution from
            which the random perturbation on x was drawn. Effective only when rand_init is
            True. Default equals to eps.
  :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
            memory or for unit tests that intentionally pass strange input)
  :return: a tensor for the adversarial example
  """
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
  if norm == 1:
    raise NotImplementedError("It's not clear that FGM is a good inner loop"
                              " step for PGD when norm=1, because norm=1 FGM "
                              " changes only one pixel at a time. We need "
                              " to rigorously test a strong norm=1 PGD "
                              "before enabling this feature.")
  if norm not in [np.inf, 2,'snr']:
    raise ValueError("Norm order must be either np.inf or 2.")
  if eps < 0:
    raise ValueError(
        "eps must be greater than or equal to 0, got {} instead".format(eps))
  if eps == 0:
    return x
  if eps_iter < 0:
    raise ValueError(
        "eps_iter must be greater than or equal to 0, got {} instead".format(eps_iter))
  if eps_iter == 0:
    return x

  assert eps_iter <= eps, (eps_iter, eps)
  if clip_min is not None and clip_max is not None:
    if clip_min > clip_max:
      raise ValueError(
          "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
              clip_min, clip_max))

  asserts = []

  # If a data range was specified, check that the input was in that range
  if clip_min is not None:
    assert_ge = torch.all(torch.ge(x, torch.tensor(clip_min, device=x.device, dtype=x.dtype)))
    asserts.append(assert_ge)

  if clip_max is not None:
    assert_le = torch.all(torch.le(x, torch.tensor(clip_max, device=x.device, dtype=x.dtype)))
    asserts.append(assert_le)

  # Initialize loop variables
  if rand_init:
    if rand_minmax is None:
      rand_minmax = eps  
    eta = torch.zeros_like(x).uniform_(-rand_minmax, rand_minmax)
    if norm=='snr':
      f = torch.Tensor(np.sqrt(np.mean(x.cpu().detach().numpy()**2,(1,2))/((10**(rand_minmax/10))*np.mean((eta.cpu().detach().numpy())**2,(1,2)))))  
      f = torch.transpose(torch.transpose(f.repeat([x.size(1),x.size(2),1]),2,1),1,0)
     
      eta*=f.to(device)
      print(np.mean(eta.cpu().detach().numpy()**2,(1,2)),np.mean(x.cpu().detach().numpy()**2,(1,2)))
      print(10*np.log10(np.mean(x.cpu().detach().numpy()**2,(1,2))/np.mean(eta.cpu().detach().numpy()**2,(1,2))))
  else:
    eta = torch.zeros_like(x)
  # Clip eta
  if norm != 'snr':
    eta = clip_eta(eta, norm, eps)
  adv_x = x + eta
  if clip_min is not None or clip_max is not None:
    adv_x = torch.clamp(adv_x, clip_min, clip_max)

  if y is None:
    # Using model predictions as ground truth to avoid label leaking
    if(True):
        if threshold is None:
            threshold = 0.5
        y = model_fn.forward(x)
        y = torch.round(y)
        y = y.detach()
        #if(model_fn(x)<threshold):
        #    y = torch.Tensor([1]).unsqueeze(0)
        #else:
        #    y = torch.Tensor([0]).unsqueeze(0)
        #y = y.to(device)


    #    _, y = torch.max(model_fn(x), 1)
  i = 0
  while i < nb_iter:
    adv_x = fast_gradient_method(model_fn, adv_x, eps_iter, norm,
                                 clip_min=clip_min, clip_max=clip_max, y=y, targeted=targeted)

    # Clipping perturbation eta to norm norm ball
    eta = adv_x - x
    eta = clip_eta(eta, norm, eps)
    adv_x = x + eta

    # Redo the clipping.
    # FGM already did it, but subtracting and re-adding eta can add some
    # small numerical error.
    if clip_min is not None or clip_max is not None:
      adv_x = torch.clamp(adv_x, clip_min, clip_max)
    i += 1

  asserts.append(eps_iter <= eps)
  if norm == np.inf and clip_min is not None:
    # TODO necessary to cast clip_min and clip_max to x.dtype?
    asserts.append(eps + clip_min <= clip_max)

  if sanity_checks:
    assert np.all(asserts)
  return adv_x


def deep_representations(model_fn, x_source, x_target, eps, eps_iter, nb_iter, norm,
        clip_min=None, clip_max=None): 
  
    #optimizer = torch.optim.SGD({x}, lr=0.5,momentum=0.9)
    #delta = torch.zeros(x_source.size, dtype=torch.float, requires_grad=False)
    #optimizer = torch.optim.LBFGS({x_source})
    x_orig = x_source
    for i in range(nb_iter):
        #optimizer.zero_grad()
        x_source = x_source.clone().detach().to(torch.float).requires_grad_(True)
        
        loss = torch.div(torch.norm(model_fn(x_source)-model_fn(x_target), dim=1), torch.norm(model_fn(x_target), dim=1))
        loss.backward()
        l = len(x_source.shape) - 1
        g = x_source.grad
        x_source.grad.zero_()
        #g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, *([1]*l))
        #scaled_g = g / (g_norm + 1e-10)
        #x_source = x_source + scaled_g * eps_iter
        optimal_perturbation = optimize_linear(g, eps, norm)
        x_source = x_source + optimal_perturbation
        print(eps, norm, optimal_perturbation)
        input('Wait')
        #loss_feat = loss_fn(model_fn(x_source),model_fn(x_target))
        #loss_input = loss_fn(x_source,x_orig)
        #print(loss_feat, loss_input)
        #loss = 100*loss_feat + loss_input
        #loss.backward()
        #optimizer.step(loss.backward())
        #optimal_perturbation = optimize_linear(x.grad,1e-3,2)
        #x = x+optimal_perturbation
        #x_source = x_source - x_source.grad * 0.01

    return x_source
