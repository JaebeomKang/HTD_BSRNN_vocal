# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Main training loop."""

import logging
import os
import sys
import gc
import time
import zipfile
      
from dora import get_xp
from dora.utils import write_and_rename
from dora.log import LogProgress, bold
import torch
import torchaudio as ta
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import librosa

from . import augment, distrib, states, pretrained
from .apply import apply_model
from .ema import ModelEMA
from .evaluate import evaluate, new_sdr
from .svd import svd_penalty
from .utils import pull_metric, EMA

logger = logging.getLogger(__name__)


def _summary(metrics):
    return " | ".join(f"{key.capitalize()}={val}" for key, val in metrics.items())


class Solver(object):
    def __init__(self, loaders, model, optimizer, args):
        self.args = args
        self.loaders = loaders

        self.model = model
        self.optimizer = optimizer
        self.quantizer = states.get_quantizer(self.model, args.quant, self.optimizer)
        self.dmodel = distrib.wrap(model)
        self.device = next(iter(self.model.parameters())).device

        # Exponential moving average of the model, either updated every batch or epoch.
        # The best model from all the EMAs and the original one is kept based on the valid
        # loss for the final best model.
        self.emas = {'batch': [], 'epoch': []}
        for kind in self.emas.keys():
            decays = getattr(args.ema, kind)
            device = self.device if kind == 'batch' else 'cpu'
            if decays:
                for decay in decays:
                    self.emas[kind].append(ModelEMA(self.model, decay, device=device))

        # data augment
        augments = [augment.Shift(shift=int(args.dset.samplerate * args.dset.shift),
                                  same=args.augment.shift_same)]
        if args.augment.flip:
            augments += [augment.FlipChannels(), augment.FlipSign()]
        for aug in ['scale', 'remix']:
            kw = getattr(args.augment, aug)
            if kw.proba:
                augments.append(getattr(augment, aug.capitalize())(**kw))
        self.augment = torch.nn.Sequential(*augments)

        xp = get_xp()
        self.folder = xp.folder
        # Checkpoints
        self.checkpoint_file = xp.folder / 'checkpoint.th'
        self.best_file = xp.folder / 'best.th'
        logger.debug("Checkpoint will be saved to %s", self.checkpoint_file.resolve())
        self.best_state = None
        self.best_changed = False

        self.link = xp.link
        self.history = self.link.history

        self._reset()

    def _serialize(self, epoch):
        package = {}
        package['state'] = self.model.state_dict()
        package['optimizer'] = self.optimizer.state_dict()
        package['history'] = self.history
        package['best_state'] = self.best_state
        package['args'] = self.args
        for kind, emas in self.emas.items():
            for k, ema in enumerate(emas):
                package[f'ema_{kind}_{k}'] = ema.state_dict()
        with write_and_rename(self.checkpoint_file) as tmp:
            torch.save(package, tmp)

        save_every = self.args.save_every
        if save_every and (epoch + 1) % save_every == 0 and epoch + 1 != self.args.epochs:
            with write_and_rename(self.folder / f'checkpoint_{epoch + 1}.th') as tmp:
                torch.save(package, tmp)

        if self.best_changed:
            # Saving only the latest best model.
            with write_and_rename(self.best_file) as tmp:
                package = states.serialize_model(self.model, self.args)
                package['state'] = self.best_state
                torch.save(package, tmp)
            self.best_changed = False

    def _reset(self):
        """Reset state of the solver, potentially using checkpoint."""
        if self.checkpoint_file.exists():
            logger.info(f'Loading checkpoint model: {self.checkpoint_file}')
            package = torch.load(self.checkpoint_file, 'cpu')
            self.model.load_state_dict(package['state'])
            self.optimizer.load_state_dict(package['optimizer'])
            self.history[:] = package['history']
            self.best_state = package['best_state']
            for kind, emas in self.emas.items():
                for k, ema in enumerate(emas):
                    ema.load_state_dict(package[f'ema_{kind}_{k}'])
        elif self.args.continue_pretrained:
            model = pretrained.get_model(
                name=self.args.continue_pretrained,
                repo=self.args.pretrained_repo)
            self.model.load_state_dict(model.state_dict())
        elif self.args.continue_from:
            name = 'checkpoint.th'
            root = self.folder.parent
            cf = root / str(self.args.continue_from) / name
            logger.info("Loading from %s", cf)
            package = torch.load(cf, 'cpu')
            self.best_state = package['best_state']
            if self.args.continue_best:
                self.model.load_state_dict(package['best_state'], strict=False)
            else:
                self.model.load_state_dict(package['state'], strict=False)
            if self.args.continue_opt:
                self.optimizer.load_state_dict(package['optimizer'])

    def _format_train(self, metrics: dict) -> dict:
        """Formatting for train/valid metrics."""
        losses = {
            'loss': format(metrics['loss'], ".4f"),
            'reco': format(metrics['reco'], ".4f"),
        }
        if 'nsdr' in metrics:
            losses['nsdr'] = format(metrics['nsdr'], ".3f")
        if self.quantizer is not None:
            losses['ms'] = format(metrics['ms'], ".2f")
        if 'grad' in metrics:
            losses['grad'] = format(metrics['grad'], ".4f")
        if 'best' in metrics:
            losses['best'] = format(metrics['best'], '.4f')
        if 'bname' in metrics:
            losses['bname'] = metrics['bname']
        if 'penalty' in metrics:
            losses['penalty'] = format(metrics['penalty'], ".4f")
        if 'hloss' in metrics:
            losses['hloss'] = format(metrics['hloss'], ".4f")
        return losses

    def _format_test(self, metrics: dict) -> dict:
        """Formatting for test metrics."""
        losses = {}
        if 'sdr' in metrics:
            losses['sdr'] = format(metrics['sdr'], '.3f')
        if 'nsdr' in metrics:
            losses['nsdr'] = format(metrics['nsdr'], '.3f')
        for source in self.model.sources:
            key = f'sdr_{source}'
            if key in metrics:
                losses[key] = format(metrics[key], '.3f')
            key = f'nsdr_{source}'
            if key in metrics:
                losses[key] = format(metrics[key], '.3f')
        return losses

    def train(self):
        # Optimizing the model
        if self.history:
            logger.info("Replaying metrics from previous run")
        for epoch, metrics in enumerate(self.history):
            formatted = self._format_train(metrics['train'])
            logger.info(
                bold(f'Train Summary | Epoch {epoch + 1} | {_summary(formatted)}'))
            formatted = self._format_train(metrics['valid'])
            logger.info(
                bold(f'Valid Summary | Epoch {epoch + 1} | {_summary(formatted)}'))
            if 'test' in metrics:
                formatted = self._format_test(metrics['test'])
                if formatted:
                    logger.info(bold(f"Test Summary | Epoch {epoch + 1} | {_summary(formatted)}"))

        epoch = 0
        for epoch in range(len(self.history), self.args.epochs):
            # Train one epoch
            self.model.train()  # Turn on BatchNorm & Dropout
            logger.info('-' * 70)
            logger.info("Training...")
            metrics = {}
            metrics['train'] = self._run_one_epoch(epoch)
            formatted = self._format_train(metrics['train'])
            logger.info(
                    bold(f'Train Summary | Epoch {epoch + 1} | {_summary(formatted)}'))

            # Cross validation
            logger.info('-' * 70)
            logger.info('Cross validation...')
            self.model.eval()  # Turn off Batchnorm & Dropout
            with torch.no_grad():
                valid = self._run_one_epoch(epoch, train=False)
                bvalid = valid
                bname = 'main'
                state = states.copy_state(self.model.state_dict())
                metrics['valid'] = {}
                metrics['valid']['main'] = valid
                key = self.args.test.metric
                for kind, emas in self.emas.items():
                    for k, ema in enumerate(emas):
                        with ema.swap():
                            valid = self._run_one_epoch(epoch, train=False)
                        name = f'ema_{kind}_{k}'
                        metrics['valid'][name] = valid
                        a = valid[key]
                        b = bvalid[key]
                        if key.startswith('nsdr'):
                            a = -a
                            b = -b
                        if a < b:
                            bvalid = valid
                            state = ema.state
                            bname = name
                    metrics['valid'].update(bvalid)
                    metrics['valid']['bname'] = bname

            valid_loss = metrics['valid'][key]
            mets = pull_metric(self.link.history, f'valid.{key}') + [valid_loss]
            if key.startswith('nsdr'):
                best_loss = max(mets)
            else:
                best_loss = min(mets)
            metrics['valid']['best'] = best_loss
            if self.args.svd.penalty > 0:
                kw = dict(self.args.svd)
                kw.pop('penalty')
                with torch.no_grad():
                    penalty = svd_penalty(self.model, exact=True, **kw)
                metrics['valid']['penalty'] = penalty

            formatted = self._format_train(metrics['valid'])
            logger.info(
                bold(f'Valid Summary | Epoch {epoch + 1} | {_summary(formatted)}'))

            # Save the best model
            if valid_loss == best_loss or self.args.dset.train_valid:
                logger.info(bold('New best valid loss %.4f'), valid_loss)
                self.best_state = states.copy_state(state)
                self.best_changed = True

            # Eval model every `test.every` epoch or on last epoch
            should_eval = (epoch + 1) % self.args.test.every == 0
            is_last = epoch == self.args.epochs - 1
            # # Tries to detect divergence in a reliable way and finish job
            # # not to waste compute.
            # # Commented out as this was super specific to the MDX competition.
            # reco = metrics['valid']['main']['reco']
            # div = epoch >= 180 and reco > 0.18
            # div = div or epoch >= 100 and reco > 0.25
            # div = div and self.args.optim.loss == 'l1'
            # if div:
            #     logger.warning("Finishing training early because valid loss is too high.")
            #     is_last = True
            if should_eval or is_last:
                # Evaluate on the testset
                logger.info('-' * 70)
                logger.info('Evaluating on the test set...')
                # We switch to the best known model for testing
                if self.args.test.best:
                    state = self.best_state
                else:
                    state = states.copy_state(self.model.state_dict())
                compute_sdr = self.args.test.sdr and is_last
                with states.swap_state(self.model, state):
                    with torch.no_grad():
                        metrics['test'] = evaluate(self, compute_sdr=compute_sdr)
                formatted = self._format_test(metrics['test'])
                logger.info(bold(f"Test Summary | Epoch {epoch + 1} | {_summary(formatted)}"))


            self.link.push_metrics(metrics)

            if distrib.rank == 0:
                # Save model each epoch
                self._serialize(epoch)
                logger.debug("Checkpoint saved to %s", self.checkpoint_file.resolve())
            if is_last:
                break

    def _run_one_epoch(self, epoch, train=True):
        args = self.args
        data_loader = self.loaders['train'] if train else self.loaders['valid']
        if distrib.world_size > 1 and train:
            data_loader.sampler.set_epoch(epoch)

        label = ["Valid", "Train"][train]
        name = label + f" | Epoch {epoch + 1}"
        total = len(data_loader)
        if args.max_batches:
            total = min(total, args.max_batches)
        logprog = LogProgress(logger, data_loader, total=total,
                              updates=self.args.misc.num_prints, name=name)
        averager = EMA()

        for idx, sources in enumerate(logprog):
            sources = sources.to(self.device)
            if train:
                sources = self.augment(sources)
                mix = sources.sum(dim=1)
                sources = sources[:, 3]        # ONLY VOCLAS
            else:
                mix = sources[:, 0]
                sources = sources[:, 1:]
                sources = sources[:, 3]

            if not train and self.args.valid_apply:
                self.model.segment = self.model.time_duration / args.dset.samplerate        # time_duration / samplerate
                estimate = apply_model(self.model, mix, split=self.args.test.split, overlap=0)
            else:
                estimate = self.dmodel(mix)
            if train and hasattr(self.model, 'transform_target'):
                sources = self.model.transform_target(mix, sources)
            assert estimate.shape == sources.shape, (estimate.shape, sources.shape)
            
            if args.optim.loss == 'l1':
                # T loss
                loss = F.l1_loss(estimate, sources)

                # F loss
                B, C, length = sources.shape
                spec_est = torch.stft(estimate.reshape(-1, length), n_fft=self.model.nfft, hop_length=self.model.hop_length, window=torch.hann_window(self.model.nfft).to(estimate),
                                        win_length=self.model.nfft, normalized=True, center=True, return_complex=True, pad_mode='reflect')
                spec_sources = torch.stft(sources.reshape(-1, length), n_fft=self.model.nfft, hop_length=self.model.hop_length, window=torch.hann_window(self.model.nfft).to(sources),
                                        win_length=self.model.nfft, normalized=True, center=True, return_complex=True, pad_mode='reflect')
                _, freqs, frame = spec_sources.shape             
                spec_est = spec_est.view(B, C, freqs, frame).contiguous()
                spec_sources = spec_sources.view(B, C, freqs, frame).contiguous()
                if spec_est.dtype == torch.cfloat:
                    spec_est = torch.view_as_real(spec_est).permute(0, 1, 4, 2, 3).contiguous()
                if spec_sources.dtype == torch.cfloat:
                    spec_sources = torch.view_as_real(spec_sources).permute(0, 1, 4, 2, 3).contiguous()

                loss += F.l1_loss(spec_est, spec_sources)

                reco = loss
            elif args.optim.loss == 'mse':
                loss = F.mse_loss(estimate, sources)
                reco = loss**0.5
            else:
                raise ValueError(f"Invalid loss {self.args.loss}")

            ms = 0
            if self.quantizer is not None:
                ms = self.quantizer.model_size()
            if args.quant.diffq:
                loss += args.quant.diffq * ms

            losses = {}
            losses['reco'] = reco
            losses['ms'] = ms

            if not train:
                nsdr = new_sdr(sources, estimate.detach()).mean(0)
                losses[f'nsdr'] = nsdr

            if train and args.svd.penalty > 0:
                kw = dict(args.svd)
                kw.pop('penalty')
                penalty = svd_penalty(self.model, **kw)
                losses['penalty'] = penalty
                loss += args.svd.penalty * penalty

            losses['loss'] = loss

            del sources, mix, estimate, reco, ms, spec_est, spec_sources
            gc.collect()
            torch.cuda.empty_cache()

            # optimize model in training mode
            if train:
                loss.backward()
                grad_norm = 0
                grads = []
                for p in self.model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm()**2
                        grads.append(p.grad.data)
                losses['grad'] = grad_norm ** 0.5
                if args.optim.clip_grad:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        args.optim.clip_grad)

                if self.args.flag == 'uns':
                    for n, p in self.model.named_parameters():
                        if p.grad is None:
                            print('no grad', n)
                self.optimizer.step()
                self.optimizer.zero_grad()
                for ema in self.emas['batch']:
                    ema.update()
            losses = averager(losses)
            logs = self._format_train(losses)
            logprog.update(**logs)
            # Just in case, clear some memory
            del loss
            gc.collect()
            torch.cuda.empty_cache()
            if args.max_batches == idx:
                break
            if self.args.debug and train:
                break
            if self.args.flag == 'debug':
                break
        if train:
            for ema in self.emas['epoch']:
                ema.update()
        return distrib.average(losses, idx + 1)


    def save_spec(self, audio, title, mel=False):
        if mel:
            S = librosa.feature.melspectrogram(y=audio, sr=self.args.dset.samplerate, n_fft=self.model.nfft, hop_length=self.model.hop_length)
            S_dB = librosa.power_to_db(S, ref=np.max)
            fig, ax = plt.subplots()
            img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=self.args.dset.samplerate, fmax=8000, ax=ax, hop_length=self.model.hop_length)
            ax.set(title=title)
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            plt.savefig(self.folder / 'spectrogram' / title)
        else:
            D = librosa.stft(audio)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            fig, ax = plt.subplots()
            img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', sr=self.args.dset.samplerate, ax=ax)
            ax.set(title=title)
            fig.colorbar(img, ax=ax, format="%+2.f dB")
            plt.savefig(self.folder / 'spectrogram' / title)
        plt.close('all')

    def get_sample(self):
        data_loader = self.loaders['valid']
        epoch = len(self.history)
        self.model.eval()
        self.model.segment = self.model.time_duration / self.args.dset.samplerate
        os.makedirs(self.folder / 'sample', exist_ok=True)
        os.makedirs(self.folder / 'spectrogram', exist_ok=True)
        spec_range = [self.args.test.spec_starts * self.args.dset.samplerate, self.args.test.spec_starts * self.args.dset.samplerate + self.model.time_duration]
        for music_num, sources in enumerate(data_loader):
            if music_num == self.args.test.activate_coord[0]:
                sources = sources.to(self.device)
                mix = sources[:, 0]
                sources = sources[:, 1:]
                sources = sources[:, 3]
                estimate = apply_model(self.model, mix, split=self.args.test.split, overlap=0)
                assert estimate.shape == sources.shape, (estimate.shape, sources.shape)
                
                for batch_num, estimate_batch in enumerate(estimate):
                    if batch_num == self.args.test.activate_coord[1]:
                        ta.save(self.folder / 'sample' / f'epoch{epoch}_music{music_num}_batch{batch_num}_mix_{self.args.test.spec_starts}_gt.wav', mix[batch_num][:,spec_range[0]:spec_range[1]].cpu(), self.args.dset.samplerate)
                        self.save_spec(mix[batch_num][0][spec_range[0]:spec_range[1]].cpu().numpy(), 
                                        f'epoch{epoch}_music{music_num}_batch{batch_num}_mix_{self.args.test.spec_starts}_gt', mel=True)
                        
                        ta.save(self.folder / 'sample' / f'epoch{epoch}_music{music_num}_batch{batch_num}_{self.model.sources[3]}_{self.args.test.spec_starts}_gt.wav', sources[batch_num][:,spec_range[0]:spec_range[1]].cpu(), self.args.dset.samplerate)
                        self.save_spec(sources[batch_num][0][spec_range[0]:spec_range[1]].cpu().numpy(), 
                                        f'epoch{epoch}_music{music_num}_batch{batch_num}_{self.model.sources[3]}_{self.args.test.spec_starts}_gt', mel=True)

                        ta.save(self.folder / 'sample' / f'epoch{epoch}_music{music_num}_batch{batch_num}_{self.model.sources[3]}_{self.args.test.spec_starts}_proposed.wav', estimate_batch[:,spec_range[0]:spec_range[1]].cpu(), self.args.dset.samplerate)
                        self.save_spec(estimate_batch[0][spec_range[0]:spec_range[1]].cpu().numpy(), 
                                        f'epoch{epoch}_music{music_num}_batch{batch_num}_{self.model.sources[3]}_{self.args.test.spec_starts}_proposed', mel=True)
                    del estimate_batch
                del sources, mix, estimate
            gc.collect()
            torch.cuda.empty_cache()

    def get_loss(self):
        os.makedirs(self.folder / 'loss plot', exist_ok=True)
        train_loss = []
        valid_loss = []
        for epoch, metrics in enumerate(self.history):
            train_loss.append(float(format(metrics['train']['loss'], ".4f")))
            valid_loss.append(float(format(metrics['valid']['loss'], ".4f")))
        x_len = range(1, len(train_loss)+1)
        plt.plot(x_len, train_loss, marker='.', label="Train-set Loss")
        plt.plot(x_len, valid_loss, marker='.', label="Validation-set Loss")
        plt.grid()
        plt.legend(loc='upper right')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(self.folder / 'loss plot' / 'loss_plot')
        plt.close('all')