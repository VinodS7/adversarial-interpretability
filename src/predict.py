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
    parser.add_argument('--dataset',
            type=str, default='jamendo',
            help='Name of the dataset to use (default: %(default)s)')
    parser.add_argument('--pitchshift', metavar='PERCENT',
            type=float, default=0.0,
            help='Perform test-time pitch-shifting of given amount and '
                 'direction in percent (e.g., -10 shifts down by 10%%).')
    parser.add_argument('--loudness', metavar='DB',
            type=float,default=0.0,
            help='Perform test-time loudness adjustment of a given anount and'
            'direction in decibel (e.g., -3 decreases volume by 3dB).')
    parser.add_argument('--adversarial', type=int,
            default=0,
            help='Predict on normal data or adversarial data')
    parser.add_argument('--mem-use',
            type=str, choices=('high', 'mid', 'low'), default='mid',
            help='How much main memory to use. More memory allows a faster '
                 'implementation, applying the network as a fully-'
                 'convolutional net to longer excerpts or the full files. '
                 '(default: %(default)s)')
    parser.add_argument('--cache-spectra', metavar='DIR',
            type=str, default=None,
            help='Store spectra in the given directory (disabled by default)')
    parser.add_argument('--plot',
            action='store_true', default=False,
            help='If given, plot each spectrogram with predictions on screen.')
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
    parser.add_argument('--input_type',default='baseline', type=str,
            help='Input type for model')
    parser.add_argument('--model_type',default='CNN', type=str,
            help='Set model architecture for use')
    return parser

def main():
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # parse command line
    parser = opts_parser()
    options = parser.parse_args()
    modelfile = options.modelfile

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

    # - create generator for spectra
    spects = (cached(options.cache_spectra and
                     os.path.join(options.cache_spectra, fn + '.npy'),
                     audio.extract_spect,
                     os.path.join(datadir, 'audio', fn),
                     sample_rate, frame_len, fps)
              for fn in filelist)
  
    # - pitch-shift if needed
    if options.pitchshift:
        import scipy.ndimage
        spline_order = 2
        spects = (scipy.ndimage.affine_transform(
                    spect, (1, 1 / (1 + options.pitchshift / 100.)),
                    output_shape=(len(spect), mel_max),
                    order=spline_order)
                  for spect in spects)

    
    if options.loudness:
        spects = (spect * float(10.**(options.loudness / 10.))
                    for spect in spects)

    # - prepare mel filterbank
    filterbank = audio.create_mel_filterbank(sample_rate, frame_len, mel_bands,
                                             mel_min, mel_max)
    filterbank = filterbank[:bin_mel_max].astype(floatX)

    # - define generator for mel spectra
    spects = (np.log(np.maximum(np.dot(spect[:, :bin_mel_max], filterbank),
                                1e-7))
              for spect in spects)

    # - load mean/std
    meanstd_file = os.path.join(os.path.dirname(__file__),
                                '%s_meanstd.npz' % options.dataset)
    with np.load(meanstd_file) as f:
        mean = f['mean']
        std = f['std']
    mean = mean.astype(floatX)
    istd = np.reciprocal(std).astype(floatX)
    # - define generator for Z-scoring
    spects = ((spect - mean) * istd for spect in spects)

    # - define generator for silence-padding
    pad = np.tile(((np.log(1e-7)-mean)*istd), (blocklen // 2, 1))
    spects = (np.concatenate((pad, spect, pad), axis=0) for spect in spects)

    # - we start the generator in a background thread (not required)

    mdl = model.CNNModel(model_type=options.model_type,input_type=options.input_type , is_zeromean=False,
            meanstd_file=meanstd_file)
    mdl.load_state_dict(torch.load(modelfile))
    mdl.to(device)
    mdl.eval()

    # run prediction loop
    print("Predicting:")
    predictions = []
    for spect in progress(spects, total=len(filelist), desc='File '):
        # naive way: pass excerpts of the size used during training
        # - view spectrogram memory as a 3-tensor of overlapping excerpts
        num_excerpts = len(spect) - blocklen + 1
        excerpts = np.lib.stride_tricks.as_strided(
                spect, shape=(num_excerpts, blocklen, spect.shape[1]),
                strides=(spect.strides[0], spect.strides[0], spect.strides[1]))
        
        # - pass mini-batches through the network and concatenate results
        if options.adversarial:
            preds = np.vstack(mdl.forward(attacks.projected_gradient_descent(
                mdl, torch.from_numpy(
                    np.transpose(excerpts[pos:pos + batchsize,:,:,
                    np.newaxis],(0,3,1,2))).to(device), cfg['eps'], cfg['eps_iter'],
                    cfg['nb_iter'], cfg['norm'], rand_init=False, targeted=False)).cpu().detach().numpy()
                for pos in range(0, num_excerpts, batchsize))
 
        else:
            preds = np.vstack(mdl.forward(torch.from_numpy(
                    np.transpose(excerpts[pos:pos + batchsize,:,:,
                    np.newaxis],(0,3,1,2))).to(device)).cpu().detach().numpy()
                for pos in range(0, num_excerpts, batchsize))
        predictions.append(preds)
    # save predictions
    print("Saving predictions")
    np.savez(outfile, **{fn: pred for fn, pred in zip(filelist, predictions)})

if __name__=="__main__":
    main()

