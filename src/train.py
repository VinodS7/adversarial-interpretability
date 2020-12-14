#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trains a neural network for singing voice detection.

For usage information, call with --help.

Author: Jan Schlüter
"""

from __future__ import print_function

import sys
import os
import io
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
floatX = np.float32

from progress import progress
from simplecache import cached
import audio
import znorm
from labels import create_aligned_targets
import attacks
import model
import augment
import config
from data_loader import DatasetLoader

def opts_parser():
    descr = "Trains a neural network for singing voice detection."
    parser = ArgumentParser(description=descr)
    parser.add_argument('modelfile', metavar='MODELFILE',
            type=str,
            help='File to save the learned weights to (.npz format)')
    parser.add_argument('--dataset',
            type=str, default='jamendo',
            help='Name of the dataset to use (default: %(default)s)')
    parser.add_argument('--augment',
            action='store_true', default=True,
            help='Perform train-time data augmentation (enabled by default)')
    parser.add_argument('--no-augment',
            action='store_false', dest='augment',
            help='Disable train-time data augmentation')
    parser.add_argument('--cache-spectra', metavar='DIR',
            type=str, default=None,
            help='Store spectra in the given directory (disabled by default)')
    parser.add_argument('--vars', metavar='FILE',
            action='append', type=str,
            default=[os.path.join(os.path.dirname(__file__), 'defaults.vars')],
            help='Reads configuration variables from a FILE of KEY=VALUE lines.'
            'Can be given multiple times, settings from later '
            'files overriding earlier ones. Will read defaults.vars, '
            'then files given here.')
    parser.add_argument('--var', metavar='KEY=VALUE',
            action='append', type=str,
            help='Set the configuration variable KEY to VALUE. Overrides '
            'settings from --vars options. Can be given multiple times')
    parser.add_argument('--validate',
            action='store_true', default=False,
            help='Monitor validation loss')
    parser.add_argument('--no-validate',
            action='store_false', dest='validate',
            help='Disable monitoring validation loss')
    parser.add_argument('--adversarial-training',
            type=int,default=0,
            help='Flag to determine whether adversarial training should occur'
            'or not.')
    parser.add_argument('--input_type',type=str,
            default='mel_spects', 
            help='input type for model, data and adversarial attacks')
    parser.add_argument('--model_type',type=str,
            default='CNN', 
            help='model architecture choice between CRNN and CNN')
    return parser

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # parse command line
    parser = opts_parser()
    options = parser.parse_args()
    modelfile = options.modelfile

    cfg = {}
    print(options.vars)
    for fn in options.vars:
        cfg.update(config.parse_config_file(fn))

    cfg.update(config.parse_variable_assignments(options.var))
    
    sample_rate = cfg['sample_rate']
    frame_len = cfg['frame_len']
    fps = cfg['fps']
    mel_bands = cfg['mel_bands']
    mel_min = cfg['mel_min']
    mel_max = cfg['mel_max']
    blocklen = cfg['blocklen']
    batchsize = cfg['batchsize']
    
    bin_nyquist = frame_len // 2 + 1
    bin_mel_max = bin_nyquist * 2 * mel_max // sample_rate

    # prepare dataset
    datadir = os.path.join(os.path.dirname(__file__),
                           os.path.pardir, 'datasets', options.dataset)
    
    meanstd_file = os.path.join(os.path.dirname(__file__),
                                '%s_meanstd.npz' % options.dataset)
 
    if(options.input_type=='audio'):
        dataloader = DatasetLoader(options.dataset, options.cache_spectra, datadir, input_type=options.input_type)
        batches = dataloader.prepare_audio_batches(sample_rate, frame_len, fps, blocklen, batchsize)
    else:
        dataloader = DatasetLoader(options.dataset, options.cache_spectra, datadir, input_type=options.input_type)
        batches = dataloader.prepare_batches(sample_rate, frame_len, fps,
            mel_bands, mel_min, mel_max, blocklen, batchsize)
    
    validation_data = DatasetLoader(options.dataset, '../ismir2015/experiments/mel_data/', datadir,
            dataset_split='valid', input_type='mel_spects')
    mel_spects_val, labels_val = validation_data.prepare_batches(sample_rate, frame_len, fps,
            mel_bands, mel_min, mel_max, blocklen, batchsize, batch_data=False)


    mdl = model.CNNModel(model_type=options.model_type, input_type=options.input_type,
            is_zeromean=False,sample_rate=sample_rate, frame_len=frame_len,
            fps=fps, mel_bands=mel_bands, mel_min=mel_min, mel_max=mel_max,
            bin_mel_max=bin_mel_max, meanstd_file=meanstd_file, device=device)
    mdl = mdl.to(device)
    
    #Setting up learning rate and learning rate parameters
    initial_eta = cfg['initial_eta']
    eta_decay = cfg['eta_decay']
    momentum = cfg['momentum']
    eta_decay_every = cfg.get('eta_decay_every', 1)
    eta = initial_eta

    #set up loss
    criterion = torch.nn.BCELoss()

    #set up optimizer
    optimizer = torch.optim.SGD(mdl.parameters(),lr=eta,momentum=momentum,nesterov=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=eta_decay_every,gamma=eta_decay)

    #set up optimizer 
    writer = SummaryWriter(os.path.join(modelfile,'runs'))

    
    epochs = cfg['epochs']
    epochsize = cfg['epochsize']
    batches = iter(batches)
    
    #conditions to save model
    best_val_loss = 100000.
    best_val_error = 1.

    for epoch in range(epochs):
        # - Initialize certain parameters that are used to monitor training
        err = 0
        total_norm = 0
        loss_accum = 0
        mdl.train(True)
        # - Compute the L-2 norm of the gradients
        for p in mdl.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        # - Start the training for this epoch
        for batch in progress(range(epochsize), min_delay=0.5,desc='Epoch %d/%d: Batch ' % (epoch+1, epochs)):
            data = next(batches)
            if(options.input_type=='audio' or options.input_type=='stft'):
                input_data = data[0]
            else:
                input_data = np.transpose(data[0][:,:,:,np.newaxis],(0,3,1,2))
            labels = data[1][:,np.newaxis].astype(np.float32)
            
            #map labels to make them softer
            if not options.adversarial_training:
                labels = (0.02 + 0.96*labels)
            optimizer.zero_grad()
            
            if(options.adversarial_training):
                mdl.train(False)
                if(options.input_type=='stft'):
                    input_data_adv = attacks.PGD(mdl, torch.from_numpy(input_data).to(device), 
                        target=torch.from_numpy(labels).to(device),
                        eps=cfg['eps'], step_size=cfg['eps_iter'], 
                        iterations=cfg['nb_iter'], use_best=True, random_start=True,
                        clip_min=0, clip_max=1e8).cpu().detach().numpy()
                else:
                    input_data_adv = attacks.PGD(mdl, torch.from_numpy(input_data).to(device), 
                        target=torch.from_numpy(labels).to(device),
                        eps=cfg['eps'], step_size=cfg['eps_iter'], 
                        iterations=cfg['nb_iter'], use_best=True, random_start=True).cpu().detach().numpy()
                
                mdl.train(True)
                optimizer.zero_grad()
                outputs = mdl(torch.from_numpy(input_data_adv).to(device))
            else:
                optimizer.zero_grad()
                outputs = mdl(torch.from_numpy(input_data).to(device))
            #input(outputs.size())
            #input(mdl.conv(torch.from_numpy(input_data).to(device)).cpu().detach().numpy().shape) 
            loss = criterion(outputs, torch.from_numpy(labels).to(device))
            loss.backward()
            optimizer.step()
            print(loss.item())
            loss_accum += loss.item()
   
        # - Compute validation loss and error if desired
        if options.validate:
            mdl.input_type = 'mel_spects'
            from eval import evaluate
            mdl.train(False) 
            val_loss = 0
            preds = []
            labs = []
            max_len = fps
            
            num_iter = 0 

            for spect, label in zip(mel_spects_val, labels_val):
                num_excerpts = len(spect) - blocklen + 1
                excerpts = np.lib.stride_tricks.as_strided(
                    spect, shape=(num_excerpts, blocklen, spect.shape[1]),
                    strides=(spect.strides[0], spect.strides[0], spect.strides[1]))
                # - Pass mini-batches through the network and concatenate results
                for pos in range(0, num_excerpts, batchsize):
                    input_data = np.transpose(excerpts[pos:pos + batchsize,:,:,np.newaxis],(0,3,1,2))
                    #if (pos+batchsize>num_excerpts):
                    #    label_batch = label[blocklen//2+pos:blocklen//2+num_excerpts,
                    #            np.newaxis].astype(np.float32)
                    #else:
                    #    label_batch = label[blocklen//2+pos:blocklen//2+pos+batchsize,
                    #            np.newaxis].astype(np.float32)
                    if (pos+batchsize>num_excerpts):
                        label_batch = label[pos:num_excerpts,
                               np.newaxis].astype(np.float32)
                    else:
                        label_batch = label[pos:pos+batchsize,
                                np.newaxis].astype(np.float32)
                    
                    pred = mdl(torch.from_numpy(input_data).to(device))
                    e = criterion(pred,torch.from_numpy(label_batch).to(device))
                    preds = np.append(preds,pred[:,0].cpu().detach().numpy())
                    labs = np.append(labs,label_batch)
                    val_loss +=e.item()
                    num_iter+=1
            mdl.input_type = options.input_type
            print("Validation loss: %.3f" % (val_loss / num_iter))
            _, results = evaluate(preds,labs)
            print("Validation error: %.3f" % (1 - results['accuracy']))
            
            if(1-results['accuracy']<best_val_error):
                torch.save(mdl.state_dict(), os.path.join(modelfile, 'model.pth'))
                best_val_loss = val_loss/num_iter
                best_val_error = 1-results['accuracy']
                print('New saved model',best_val_loss, best_val_error)
                    
        #Update the learning rate
        scheduler.step()
        
        print('Training Loss per epoch', loss_accum/epochsize) 
        
        # - Save parameters for examining
        writer.add_scalar('Training Loss',loss_accum/epochsize,epoch)
        writer.add_scalar('Validation loss', val_loss/num_iter,epoch)
        writer.add_scalar('Gradient norm', total_norm, epoch)
        writer.add_scalar('Validation error', 1-results['accuracy'])
        #for param_group in optimizer.param_groups:
            #print(param_group['lr'])
    
    if not options.validate:
        torch.save(mdl.state_dict(), os.path.join(modelfile, 'model.pth'))
    with io.open(os.path.join(modelfile, 'model.vars'), 'w') as f:
        f.writelines('%s=%s\n' % kv for kv in cfg.items())

if __name__=='__main__':
    main()
