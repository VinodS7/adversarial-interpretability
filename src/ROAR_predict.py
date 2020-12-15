#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Computes predictions with a neural network trained for singing voice detection.

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
floatX = np.float32

from progress import progress
from simplecache import cached
import audio
import model
import augment
import config
import attacks
from data_loader import DatasetLoader
def opts_parser():
    descr = ("Computes predictions with a neural network trained for singing "
             "voice detection.")
    parser = ArgumentParser(description=descr)
    parser.add_argument('modelfile', metavar='MODELFILE',
            type=str,
            help='File to load the learned weights from (.npz format)')
    parser.add_argument('outfile', metavar='OUTFILE',
            type=str,
            help='File to save the prediction curves to (.npz format)')
    parser.add_argument('--lossgradient',
            type=str, default='None',
            help='Model file for loss gradient computation')
    parser.add_argument('--dataset',
            type=str, default='jamendo',
            help='Name of the dataset to use (default: %(default)s)')
    parser.add_argument('--cache-spectra', metavar='DIR',
            type=str, default=None,
            help='Store spectra in the given directory (disabled by default)')
    parser.add_argument('--loss-grad-save', metavar='DIR',
            type=str, default=None,
            help='Store loss gradients in given directory')
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
    parser.add_argument('--input_type',type=str,
            default='mel_spects', 
            help='input type for model, data and adversarial attacks')
    parser.add_argument('--model_type',default='baseline', type=str,
            help='Set model type for use')
    parser.add_argument('--ROAR', default=1, type=int,
            help='Remove highest gradients or lowest')
    parser.add_argument('--lossgrad_algorithm', type=str,
            default='grad',help='Choose algorithm for doing the'
            'loss gradient analysis')
    return parser

def main():
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # parse command line
    parser = opts_parser()
    options = parser.parse_args()
    modelfile = options.modelfile
    lossgradient = options.lossgradient

    cfg = {}
    for fn in options.vars:
        cfg.update(config.parse_config_file(fn))

    cfg.update(config.parse_variable_assignments(options.var))

    outfile = options.outfile
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
    print("Preparing data reading...")
    datadir = os.path.join(os.path.dirname(__file__),
                           os.path.pardir, 'datasets', options.dataset)

    # - load filelist
    with io.open(os.path.join(datadir, 'filelists', 'valid')) as f:
        filelist = [l.rstrip() for l in f if l.rstrip()]
    with io.open(os.path.join(datadir, 'filelists', 'test')) as f:
        filelist += [l.rstrip() for l in f if l.rstrip()]
 
    # - load mean/std
    meanstd_file = os.path.join(os.path.dirname(__file__),
                                '%s_meanstd.npz' % options.dataset)
    
    dataloader = DatasetLoader(options.dataset, options.cache_spectra, datadir, input_type=options.input_type,filelist=filelist)
    mel_spects, labels = dataloader.prepare_batches(sample_rate, frame_len, fps,
            mel_bands, mel_min, mel_max, blocklen, batchsize, batch_data=False)
    
    with np.load(meanstd_file) as f:
        mean = f['mean']
        std = f['std']
    mean = mean.astype(floatX)
    istd = np.reciprocal(std).astype(floatX)

    mdl = model.CNNModel(input_type='mel_spects_norm', is_zeromean=False,
            meanstd_file=meanstd_file, device=device)
    mdl.load_state_dict(torch.load(modelfile))
    mdl.to(device)
    mdl.eval()

    if(lossgradient!='None'):
        mdl_lossgrad =  model.CNNModel(input_type=options.input_type,
                is_zeromean=False, sample_rate=sample_rate, frame_len=frame_len, fps=fps,
                mel_bands=mel_bands, mel_min=mel_min, mel_max=mel_max,
                bin_mel_max=bin_mel_max, meanstd_file=meanstd_file, device=device)
        mdl_lossgrad.load_state_dict(torch.load(lossgradient))
        mdl_lossgrad.to(device)
        mdl_lossgrad.eval()
        criterion = torch.nn.BCELoss()
        loss_grad_val = dataloader.prepare_loss_grad_batches(options.loss_grad_save, mel_spects,
                labels, mdl_lossgrad, criterion, blocklen, batchsize, device)
 
    


    # run prediction loop
    print("Predicting:")
    predictions = []
    #for spect, g in zip(mel_spects, loss_grad_val):
    c = 0
    for spect in progress(mel_spects, total=len(filelist), desc='File '):
        if(lossgradient!='None'):
            g = loss_grad_val[c]
        c+=1
    # naive way: pass excerpts of the size used during training
        # - view spectrogram memory as a 3-tensor of overlapping excerpts
        num_excerpts = len(spect) - blocklen + 1
        excerpts = np.lib.stride_tricks.as_strided(
                spect.astype(floatX), shape=(num_excerpts, blocklen, spect.shape[1]),
                strides=(spect.strides[0], spect.strides[0], spect.strides[1]))
        preds = np.zeros((num_excerpts,1))
        count = 0
        for pos in range(0, num_excerpts, batchsize):
            input_data = np.transpose(excerpts[pos:pos + batchsize,:,:,np.newaxis],(0,3,1,2))
            input_data = (input_data-mean)*istd
            if lossgradient!='None':
                for i in range(input_data.shape[0]):
                    if(options.lossgrad_algorithm=='grad'):
                        rank_matrix = np.abs(g[i+pos])
                    elif(options.lossgrad_algorithm=='gradxinp'):
                        rank_matrix = np.squeeze(g[i+pos]*input_data[i,:,:,:])
                    elif(options.lossgrad_algorithm=='gradorig'):
                        rank_matrix = g[i+pos]
                    if(options.ROAR==1):
                        v = np.argsort(rank_matrix, axis=None)[-cfg['occlude']:]
                    else:
                        v = np.argsort(rank_matrix, axis=None)[:cfg['occlude']]
                    input_data[i,:,v//80,v%80] = 0
            else:
                for i in range(input_data.shape[0]):
                    #print('random')
                    v = np.random.choice(115*80, cfg['occlude'], replace=False)
                    input_data[i,:,v//80,v%80] = 0

            count+=1
            
            #print('Here')
            #preds = np.vstack(mdl.forward(torch.from_numpy(
            #            np.transpose(excerpts[pos:pos + batchsize,:,:,
            #            np.newaxis],(0,3,1,2))).to(device)).cpu().detach().numpy()
            #        for pos in range(0, num_excerpts, batchsize))
            
            preds[pos:pos+batchsize,:] = mdl(torch.from_numpy(input_data).to(device)).cpu().detach().numpy()
        print('Here')
        predictions.append(preds)
    # save predictions
    print("Saving predictions")
    np.savez(outfile, **{fn: pred for fn, pred in zip(filelist, predictions)})

if __name__=="__main__":
    main()

