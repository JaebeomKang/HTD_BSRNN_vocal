# HTD_BSRNN(vocal)

*Graduation project at Yonsei Univ.*

## Abstract

  This project is about music source separation (MSS), one of the audio signal processing tasks. We analyze the trend of MSS algorithms and focus on the hybrid domain approach of Hybrid Transformer Demucs (HTDemucs), the current state-of-the-art (SOTA) model. HTDemucs is an end-to-end hybrid source separation model that combines the traditional two mainstream approaches which are the waveform-domain algorithm and the spectrogram-domain algorithm by using cross-domain attention, which has the advantage of taking the strength of both the time-domain approach and the time-frequency (TF)-domain approach. However, contrary to our expectations, when we actually examined the output of each individual branch, we found that only one side of the branch actually works. And we inferred that these results make the SDR performance saturate more early.

  Based on these findings, we proposed the need to improve one branch of this model, the spectrogram-based approach, and applied Band-Split RNN (BSRNN), which models spectrograms by splitting them into pre-determined frequency bands, as a solution. In order to solve the new initial phase inconsistency problem that may arise when applying these modifications, we trained with spectrogram reconstruction loss in addition to existing direct reconstruction loss on waveforms. With these approaches, we observed that each branch worked properly(Time branch: Capture high-frequency characteristics better, Spectral branch: Capture low-frequency characteristics better). By these facts, we concluded that we mitigated the problem of imbalanced branch performance. And we also observed an increase in SDR score, a performance metric in MSS task, compared to HTDemucs, with even smaller model size.

**Key words** : *Music Source Separation, Hybrid-domain approach, Band-split approach*



## Introduction

<img src="C:\Users\jaebo\AppData\Roaming\Typora\typora-user-images\image-20230823140805323.png" alt="image-20230823140805323" style="zoom: 50%;" />

Music Source Separation (MSS) is a task that extracts sources[stems] such as Vocals, Bass, Drums, etc from a mixture of music inputs. It has wide applications such as music unmixing, and music information retrieval (MIR). Also separated sources can be reused for different versions of the song and located in different spatial positions. Additionally, it has scalability in entertainment and education fields.

The challenge is that we need to be able to extract only the target stem components, even when there are more components from other stems compared to the target stem. However, the “Cocktail party effect” [1] describes how the human brain is able to separate a single conversation out of surrounding noise from a room full of people chatting. Inspired by this effect, many traditional algorithms have been designed to estimate the mask, which can simply filter out the target stem from mixture input. Another challenge is that music signals are trickier to handle than speech signals. Music signals are generally made at higher sample rates than speech and have super wide-band compared to speech signals, so they are known to have more constraints than speech signals. Even singing voice [vocals] and speech have different characteristics in terms of fundamental frequency, loudness, formants, etc. [2]. 

Since the 2015 Signal Separation Evaluation Campaign (SiSEC) [3], the MSS community has mainly put forward deep learning models. And by the birth of the reference dataset used for the MSS benchmark consisting of 150 songs in two versions: HQ and non-HQ, called MUSDB18 [4] [5], the research was spurred. Its training set consists of 87 songs. Early MSS models had developed by citing many algorithms used in speech enhancement or speech separation. Most work has focused on training supervised models that separate songs into four stems: drums, bass, vocals, and others (all other instruments). 

There are two main approaches to deep learning models: waveform-based approaches and spectrogram-based approaches. And recently, a hybrid approach[6] that combines the two domains has been gaining traction. The state-of-the-art model, Hybrid transformer demucs using Cross-Domain Transformer is one of them. In this work, we found that HTDemucs did not have the full benefits of hybrid approaches and were imbalanced performing on one domain branch, so we attempted to address this by replacing the existing spectrogram-domain branch with Band-split RNN [7]. We also observe the SDR, Spectrogram, and human perception to verify the performance.
