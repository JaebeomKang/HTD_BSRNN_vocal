# HTD_BSRNN(vocal)

Graduation project at Yonsei Univ.

## Abstract

  This paper is about music source separation (MSS), one of the audio signal processing tasks. We analyze the trend of MSS algorithms and focus on the hybrid domain approach of Hybrid Transformer Demucs (HTDemucs), the current state-of-the-art (SOTA) model. HTDemucs is an end-to-end hybrid source separation model that combines the traditional two mainstream approaches which are the waveform-domain algorithm and the spectrogram-domain algorithm by using cross-domain attention, which has the advantage of taking the strength of both the time-domain approach and the time-frequency (TF)-domain approach. However, contrary to our expectations, when we actually examined the output of each individual branch, we found that only one side of the branch actually works. And we inferred that these results make the SDR performance saturate more early.

  Based on these findings, we proposed the need to improve one branch of this model, the spectrogram-based approach, and applied Band-Split RNN (BSRNN), which models spectrograms by splitting them into pre-determined frequency bands, as a solution. In order to solve the new initial phase inconsistency problem that may arise when applying these modifications, we trained with spectrogram reconstruction loss in addition to existing direct reconstruction loss on waveforms. With these approaches, we observed that each branch worked properly(Time branch: Capture high-frequency characteristics better, Spectral branch: Capture low-frequency characteristics better). By these facts, we concluded that we mitigated the problem of imbalanced branch performance. And we also observed an increase in SDR score, a performance metric in MSS task, compared to HTDemucs, with even smaller model size.



## Introduction

