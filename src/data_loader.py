from __future__ import print_function

import sys
import os
import io

import numpy as np
import torch
import torch.nn as nn

from progress import progress
from simplecache import cached
import audio
import znorm
from labels import create_aligned_targets
import augment
import config
import matplotlib
import matplotlib.pyplot as plt

floatX = np.float32

class DatasetLoader():
    def __init__(self, dataset, featuredir, datadir, dataset_split='train', augment=False, input_type='mel_spects', filelist=None):
        if(filelist is None):
            with io.open(os.path.join(datadir,'filelists', dataset_split)) as f:
                self.filelist = [l.rstrip() for l in f if l.rstrip()]
        else:
            self.filelist = filelist
        self.datadir = datadir
        self.dataset = dataset
        self.featuredir = featuredir
        self.augment = augment
        self.input_type = input_type
        self.dataset_split = dataset_split
        return

    def prepare_batches(self, sample_rate, frame_len, fps,
            mel_bands, mel_min, mel_max, blocklen, batchsize, batch_data=True):
        
        bin_nyquist = frame_len // 2 + 1
        bin_mel_max = bin_nyquist * 2 * mel_max // sample_rate
        spects = []
        for fn in progress(self.filelist, 'File'):
            cache_fn = (self.featuredir and
                    os.path.join(self.featuredir,fn + '.npy'))
            spects.append(cached(cache_fn, audio.extract_spect,
                os.path.join(self.datadir, 'audio', fn),
                sample_rate, frame_len, fps))
         # - load and convert corresponding labels
        print("Loading labels...")
        labels = []
        for fn, spect in zip(self.filelist, spects):
            fn = os.path.join(self.datadir, 'labels', fn.rsplit('.', 1)[0] + '.lab')
            with io.open(fn) as f:
                segments = [l.rstrip().split() for l in f if l.rstrip()]
            segments = [(float(start), float(end), label == 'sing')
                    for start, end, label in segments]
            timestamps = np.arange(len(spect)) / float(fps)
            labels.append(create_aligned_targets(segments, timestamps, np.bool))

        if(self.input_type=='stft'):
            print('Create dataset with stft output')
            if (batch_data):
                batches = augment.grab_random_excerpts(
                        spects, labels, batchsize, blocklen)
                return batches
            else:
                return spects, labels
          
        
        if (self.input_type=='mel_spects' or self.input_type=='mel_spects_norm'):
            # - prepare mel filterbank
            filterbank = audio.create_mel_filterbank(sample_rate, frame_len, mel_bands,
                                             mel_min, mel_max)
            filterbank = filterbank[:bin_mel_max].astype(floatX)

            # - precompute mel spectra, if needed, otherwise just define a generator
            mel_spects = (np.log(np.maximum(np.dot(spect[:, :bin_mel_max], filterbank),
                                    1e-7))
                  for spect in spects)
   

            if not self.augment:
                mel_spects = list(mel_spects)
                del spects

            # - load mean/std or compute it, if not computed yet
            meanstd_file = os.path.join(os.path.dirname(__file__),
                                '%s_meanstd.npz' % self.dataset)
            try:
                with np.load(meanstd_file) as f:
                    mean = f['mean']
                    std = f['std']
            except (IOError, KeyError):
                print("Computing mean and standard deviation...")
                mean, std = znorm.compute_mean_std(mel_spects)
                np.savez(meanstd_file, mean=mean, std=std)
            mean = mean.astype(floatX)
            istd = np.reciprocal(std).astype(floatX)
          
        # - prepare training data generator
        print("Preparing training data feed...")
        if not self.augment:
            # Without augmentation, we just precompute the normalized mel spectra
            # and create a generator that returns mini-batches of random excerpts
            if (self.input_type=='mel_spects'):
                print('Creating batches of mel spects without znorm')
                if (batch_data):
                    batches = augment.grab_random_excerpts(
                        mel_spects, labels, batchsize, blocklen)
                else:
                    pad = np.tile((np.log(1e-7).astype(floatX)), (blocklen//2, 80))
                    mel_spects = (np.concatenate((pad, spect, pad), axis=0) for spect in mel_spects)
                    mel_spects = list(mel_spects)
                    return mel_spects, labels
            elif (self.input_type =='mel_spects_norm'):
                print('Creating batches of mel spects with znorm')
                #mel_spects = [(spect - mean) * istd for spect in mel_spects]
                if (batch_data):
                    mel_spects = [(spect - mean) * istd for spect in mel_spects]
                    batches = augment.grab_random_excerpts(
                        mel_spects, labels, batchsize, blocklen)
                else:
                    #pad = np.tile((np.log(1e-7)-mean)*istd, (blocklen//2, 1))
                    #input(pad)
                    pad = np.tile((np.log(1e-7).astype(floatX)), (blocklen//2, 80))
                    mel_spects = (np.concatenate((pad, spect, pad), axis=0) for spect in mel_spects)
                    mel_spects = [(spect - mean) * istd for spect in mel_spects]
                    #input(mean.shape)
                    #mel_spects = [(spect - np.zeros(80)) * np.ones(80) for spect in mel_spects]
                
 
                    mel_spects = list(mel_spects)
                    
                    return mel_spects, labels
            else:
                print('Creating batches of stfts')
                batches = augment.grab_random_excerpts(
                    spects, labels, batchsize, blocklen)
        else:
            # For time stretching and pitch shifting, it pays off to preapply the
            # spline filter to each input spectrogram, so it does not need to be
            # applied to each mini-batch later.
            spline_order = cfg['spline_order']
            if spline_order > 1:
                from scipy.ndimage import spline_filter
                spects = [spline_filter(spect, spline_order).astype(floatX)
                      for spect in spects]

            # We define a function to create the mini-batch generator. This allows
            # us to easily create multiple generators for multithreading if needed.
            def create_datafeed(spects, labels):
                # With augmentation, as we want to apply random time-stretching,
                # we request longer excerpts than we finally need to return.
                max_stretch = cfg['max_stretch']
                batches = augment.grab_random_excerpts(
                    spects, labels, batchsize=batchsize,
                    frames=int(blocklen / (1 - max_stretch)))

                # We wrap the generator in another one that applies random time
                # stretching and pitch shifting, keeping a given number of frames
                # and bins only.
                max_shift = cfg['max_shift']
                batches = augment.apply_random_stretch_shift(
                    batches, max_stretch, max_shift,
                    keep_frames=blocklen, keep_bins=bin_mel_max,
                    order=spline_order, prefiltered=True)

                # We transform the excerpts to mel frequency and log magnitude.
                batches = augment.apply_filterbank(batches, filterbank)
                batches = augment.apply_logarithm(batches)

                # We apply random frequency filters
                max_db = cfg['max_db']
                batches = augment.apply_random_filters(batches, filterbank,
                                                   mel_max, max_db=max_db)

                # We apply normalization
                batches = augment.apply_znorm(batches, mean, istd)

                return batches

            # We start the mini-batch generator and augmenter in one or more
            # background threads or processes (unless disabled).
            bg_threads = cfg['bg_threads']
            bg_processes = cfg['bg_processes']
            if not bg_threads and not bg_processes:
                # no background processing: just create a single generator
                batches = create_datafeed(spects, labels)
            elif bg_threads:
                # multithreading: create a separate generator per thread
                batches = augment.generate_in_background(
                    [create_datafeed(spects, labels)
                     for _ in range(bg_threads)],
                    num_cached=bg_threads * 5)
            elif bg_processes:
                # multiprocessing: single generator is forked along with processes
                batches = augment.generate_in_background(
                    [create_datafeed(spects, labels)] * bg_processes,
                    num_cached=bg_processes * 25,
                    in_processes=True)


        return batches

    def prepare_audio_batches(self, sample_rate, frame_len, fps, blocklen, batchsize, batch_data=True):
        
        spects = []
        for fn in progress(self.filelist, 'File'):
            cache_fn = (self.featuredir and
                    os.path.join(self.featuredir,fn + '.npy'))
            spects.append(cached(cache_fn, audio.read_ffmpeg,
                os.path.join(self.datadir, 'audio', fn),
                sample_rate))

        # - load and convert corresponding labels
        print("Loading labels...")
        labels = []
        for fn, spect in zip(self.filelist, spects):
            fn = os.path.join(self.datadir, 'labels', fn.rsplit('.', 1)[0] + '.lab')
            with io.open(fn) as f:
                segments = [l.rstrip().split() for l in f if l.rstrip()]
            segments = [(float(start), float(end), label == 'sing')
                    for start, end, label in segments]
            timestamps = np.arange(len(spect)) / float(sample_rate)
            labels.append(create_aligned_targets(segments, timestamps, np.bool))
        

        if (batch_data):
            batches = augment.grab_random_audio_excerpts(
                spects, labels, batchsize, sample_rate, frame_len, fps, blocklen)
            return batches
        else:
            return spects, labels
        
    def prepare_loss_grad_batches(self, loss_grad_path, mel_spects, labels, mdl_fn, criterion, blocklen,
            batchsize, device):
        

        def loss_grad(model_fn, x, label, criterion):
            x = x.clone().detach().requires_grad_(True)
            loss = criterion(model_fn(x), label)
            grad, = torch.autograd.grad(loss, [x])
            return grad.cpu().detach().numpy()
        def compute_grads(spect, label, mdl_fn, criterion,
                blocklen, batchsize, device):      
            num_excerpts = len(spect) - blocklen + 1
            excerpts = np.lib.stride_tricks.as_strided(
                    spect, shape=(num_excerpts, blocklen, spect.shape[1]),
                    strides=(spect.strides[0], spect.strides[0], spect.strides[1]))
            total_grads = [] 
            for pos in range(0, num_excerpts, batchsize):
                input_data = np.transpose(excerpts[pos:pos + batchsize,:,:,np.newaxis],(0,3,1,2))
                if (pos+batchsize>num_excerpts):
                    label_batch = label[pos:num_excerpts,
                        np.newaxis].astype(np.float32)
                else:
                    label_batch = label[pos:pos+batchsize,
                                np.newaxis].astype(np.float32)      
                input_data_loss = input_data  
                grads = loss_grad(mdl_fn, torch.from_numpy(input_data_loss).to(device).requires_grad_(True), torch.from_numpy(label_batch).to(device), criterion)
                total_grads.append(np.squeeze(grads))
            return total_grads

        grad = []
        for spect, label, fn in zip(mel_spects, labels, self.filelist):
            print(fn)
            cache_fn = (loss_grad_path and os.path.join(loss_grad_path, fn + '.npy'))
            grad.append(cached(cache_fn, compute_grads, spect, label, mdl_fn,
                criterion, blocklen, 1, device))
            
        #for spect, label, g in zip(mel_spects, labels, grad):     
        #    num_excerpts = len(spect) - blocklen + 1
        #    print(num_excerpts)
        #    input(len(g))
        #    excerpts = np.lib.stride_tricks.as_strided(
        #            spect, shape=(num_excerpts, blocklen, spect.shape[1]),
        #            strides=(spect.strides[0], spect.strides[0], spect.strides[1]))
        #    total_grads = []  
        #    for pos in range(0, num_excerpts, batchsize):
        #        input_data = np.transpose(excerpts[pos:pos + batchsize,:,:,np.newaxis],(0,3,1,2))
        #        if (pos+batchsize>num_excerpts):
        #            label_batch = label[blocklen//2+pos:blocklen//2+num_excerpts,
        #                np.newaxis].astype(np.float32)
        #        else:
        #            label_batch = label[blocklen//2+pos:blocklen//2+pos+batchsize,
        #                        np.newaxis].astype(np.float32)      
        #        input_data_loss = input_data  
       #         grads = loss_grad(mdl_fn, torch.from_numpy(input_data_loss).to(device).requires_grad_(True), torch.from_numpy(label_batch).to(device), criterion)
       #         for i in range(batchsize):
       #             #print('not random')
       #             v1 = np.argsort(np.abs(grads[i]), axis=None)[-20:]
       #            
       #             v2 = np.argsort(np.abs(g[i+pos]),axis=None)[-20:]
       #             print(v2-v1)
#
       #         input('wait')
        return grad
        

