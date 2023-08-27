# Vocal Separation with Hybrid and Band-Split approaches

*Graduation project by Jaebeom Kang(Yonsei Univ.) & Wooseok Shin(Yonsei Univ.)*

*This paper was submitted to Yonsei University on June 14th in 2023.*

## Abstract

  This project is about music source separation (MSS), one of the audio signal processing tasks. We briefly reviewed the trend of MSS algorithms and focus on the hybrid domain approach of Hybrid Transformer Demucs (HTDemucs), the current state-of-the-art (SOTA) model. HTDemucs is an end-to-end hybrid source separation model that combines the traditional two mainstream approaches which are the waveform-domain algorithm and the spectrogram-domain algorithm by using cross-domain attention. By doing so it has the advantage of taking the strength of both the time-domain approach and the time-frequency (TF)-domain approach. However, contrary to our expectations, when we actually examined the output of each individual branch, we found that only one side of the branch actually works. And we inferred that these results make the SDR performance saturate more early.

  Based on these findings, we proposed the need to improve the spectral branch of this model and replaced it with the Band-Split RNN (BSRNN), which models spectrograms by splitting them into pre-determined frequency bands, as a solution. In order to solve the initial phase inconsistency problem that may arise when applying these modifications, we set the objective function as spectrogram reconstruction loss in addition to existing direct reconstruction loss on waveforms. With these approaches, we observed that each branch worked properly(Time branch: Capture high-frequency characteristics better, Spectral branch: Capture low-frequency characteristics better). By these facts, we concluded that we mitigated the problem of imbalanced branch performance. And we also observed an increase in SDR score, a performance metric in MSS task, compared to HTDemucs, with even smaller model size.

**Key words** : *Music Source Separation, Hybrid-domain approach, Band-split approach*

## Contents

1. Introduction

2. Related Works

   2.1. Time-Frequency domain Approaches

   2.2. Time domain Approaches

   2.3. Hybrid domain Approaches

3. Baseline

   3.1. Band-Split RNN(BSRNN)

   3.2. Hybrid Transformer Demucs(HTDemucs)

4. Proposed Work

   4.1. Imbalanced branch problems in HTDemucs

   4.2. Proposal

5. Experiments

   5.1. Setup

   5.2. Results

6. Conclusions & Discussions

P.S.) *There are quite large amounts for「2. Related Works」.  And most of them were omitted in the actual submission. It's honestly for my note-taking purposes, so if you want to, you can skip it.*

## 1. Introduction

Music Source Separation (MSS) is a task that extracts sources[stems] such as Vocals, Bass, Drums, etc from a mixture of music inputs. It has wide applications such as music unmixing, and music information retrieval (MIR). Also separated sources can be reused for different versions of the song and located in different spatial positions. Additionally, it has scalability in entertainment and education fields.

The challenge is that we need to be able to extract only the target stem components, even when there are more components from other stems compared to the target stem. However, the “Cocktail party effect” [1] describes how the human brain is able to separate a single conversation out of surrounding noise from a room full of people chatting. Inspired by this effect, many traditional algorithms have been designed to estimate the mask, which can simply filter out the target stem from mixture input. Another challenge is that music signals are trickier to handle than speech signals. Music signals are generally made at higher sample rates than speech and have super wide-band compared to speech signals, so they are known to have more constraints than speech signals. Even singing voice [vocals] and speech have different characteristics in terms of fundamental frequency, loudness, formants, etc. [2]. 

Since the 2015 Signal Separation Evaluation Campaign (SiSEC) [3], the MSS community has mainly put forward deep learning models. And by the birth of the reference dataset used for the MSS benchmark consisting of 150 songs in two versions: HQ and non-HQ, called MUSDB18 [4] [5], the research was spurred. Its training set consists of 87 songs. Early MSS models had developed by citing many algorithms used in speech enhancement or speech separation. Most work has focused on training supervised models that separate songs into four stems: drums, bass, vocals, and others (all other instruments). 

There are two main approaches to deep learning models: waveform-based approaches and spectrogram-based approaches. And recently, a hybrid approach[6] that combines the two domains has been gaining traction. The state-of-the-art model, Hybrid transformer demucs using Cross-Domain Transformer is one of them. In this work, we found that HTDemucs did not have the full benefits of hybrid approaches and were imbalanced performing on one domain branch, so we attempted to address this by replacing the existing spectrogram-domain branch with Band-split RNN [7]. We also observe the SDR, Spectrogram, and human perception to verify the performance.



## 2. Related Works

### 2.1. Time-Frequency domain Approaches

As mentioned above, there are largely two approaches to the MSS task: spectrogram-based approaches and waveform-based approaches.  Spectrogram-based approaches, also known as Time-Frequency (TF) domain approaches, use the Short-Time Fourier Transform (STFT), which is used in many areas of audio processing, and its magnitude is called the spectrogram. This can be used to effectively analyze the changes in the various frequency components over time. In the conventional approach, before deep learning was used in MSS, the spectrogram was used to solve the problem.

#### 2.1.1. Non-negative Matrix Factorization(NMF)

NMF is an algorithm that decomposes a non-negative matrix $\textbf{V}$ into non-negative factors $\textbf{B}$, $\textbf{W}$, which is $\textbf{V}\approx \textbf{BW}$. NMF is known to be effective when analyzing multivariate data.

We can observe that each column vector $\textbf{v}$ in $\textbf{V}$ is represented by $\textbf{v}=\textbf{Bw}$, i.e., it can be approximated by a linear combination of the column vectors of $\textbf{B}$ (basis vectors). Therefore, a good approximation means that $\textbf{B}$ contains basis vectors that represent the characteristics of the data well. $\textbf{B}$ and $\textbf{W}$ are obtained by finding a solution that minimizes the cost function between $\textbf{V}$ and $\textbf{BW}$, and the cost function is mainly implemented in audio signal processing by using Kullback-Leibler divergence (KL divergence) and finding a solution using iterative update[9].

$\textbf{S}$ <sub>$train$</sub>$\approx \textbf{B}$<sub>$speech$</sub> $\textbf{W}$ <sub>$speech$</sub>

$\textbf{M}$ <sub>$train$</sub>$\approx \textbf{B}$<sub>$music$</sub> $\textbf{W}$ <sub>$music$</sub>

As above, firstly obtain the basis $\textbf{B}$<sub>$speech$</sub> and $\textbf{B}$<sub>$music$</sub> of each speech spectrogram ($\textbf{S}$) and music spectrogram ($\textbf{M}$) through training using the ground truth, and then redefine the mixture spectrogram ($\textbf{X}$) as below.-

$\textbf{x}\approx [ \textbf{B}$<sub>$speech$</sub> $\textbf{B}$<sub>$music$</sub> $] \textbf{W}$

We can train only on $\textbf{W}$ above and use the weight matrix  $\textbf{W}$ <sub>$speech$</sub>, $\textbf{W}$ <sub>$music$</sub> to get the masks of speech and music as shown below.

$\widetilde{\textbf{S}}=\textbf{B}$<sub>$speech$</sub>$\textbf{W}$<sub>$S$</sub>

The NMF method separates speech and music by applying Inverse Short-Time Fourier Transform(ISTFT) after masking the obtained mask to the mixture spectrogram.
However, these approaches have the problem that it can effectively model the stationary signals, but not non-stationary signals, because each source is characterized using a single spectral basis.
There is also a PCA (Principal Component Analysis) method that assumes that music accompaniment exists in a low-rank subspace and vocal has high sparsity and performs separation [10], but it also has the same problem as NMF.

#### 2.1.2. Ideal Ratio Mask (IRM) Estimation

![IRM](/images/IRM.png)

**[Figure. 1]** Common frameworks of IRM estimation using Deep Learning.

After the success of Deep Neural Network (DNN), many attempts have been made to use Deep Learning in audio signal processing, and one of them was to estimate Ideal Ratio Mask (IRM) for MSS tasks. 

Computational Auditory Scene Analysis (CASA) refers to research that computationally models the human auditory system (cocktail party effect [1]) that extracts desired sound sources from acoustic mixtures. The main concept of CASA is the Ideal Binary Mask (IBM) [11], which refers to binary time-frequency (T-F) masking for extracting target sounds.

This performs binary masking by taking values as they are when the energy of the target signal is greater than that of the interference signal and masking it as 0 otherwise. Therefore, the value of the mask means the estimated spectral value of the target source. IBM was widely used because at that time it was believed that IBM was optimal masking that maximized Signal-to-noise ratio (SNR), which was a representative metric for measuring separation performance.

However, it was discovered that if there is an overlap when calculating STFT’s T-F unit, IBM may not be optimal for increasing SNR due to the nonlinearity of SNR due to the boundary effect [12]. To solve this problem, an Ideal Ratio Mask (IRM), a smooth masking of IBM, was proposed. The Ideal Ratio Mask formula can be seen as similar in form to the square-root Wiener filter, which is an optimal estimation of the power spectrum.

By learning such masks through DNNs, approaches that increase SNR were mainstream methods used in MSS in the early of applying deep learning [13] [14] [15] [16]. Various strategies have been developed but most of them follow similar frameworks as Figure 1. After obtaining the STFT of the signal and splitting it into Magnitude and Phase to make a real element so that networks such as DNN/RNN/CNN can learn. Then Soft Time-Frequency Mask is learned for the magnitude spectrogram and separation is performed by combining the masked magnitude spectrogram and input mixture phase. There were various attempts such as DNN [17], RNN [15], CNN [18], U-net [19], and GAN [20] as model architecture.

However, these approaches had some limitations. First of all, performing an ISTFT using the phases of the mixture makes an upper bound on the performance of SDR. Also ‘estimated’ ratio mask learned through DNN doesn't guarantee that it could increase SNR unlike the ‘ideal’ ratio mask and it was uncertain whether this learning method could generalize well even for unseen audio input [18]. Various methods were attempted to solve these limitations such as complex Ideal Ratio Mask (cIRM) [21] or Phase Sensitive Mask (PSM) [22] were attempted. In addition, the Optimal Ratio Mask (ORM) [23] method to improve IRM was introduced. Embedding space such as a deep-attractor network [24] also appeared in the masking approach. However, more than anything else, Spectrogram-based approaches have hyperparameter-sensitive problems due to the TF resolution problems that are induced by the choice of hop length, window length, etc.

### 2.2. Time domain approaches

As mentioned above, there are largely two approaches to the MSS task: spectrogram-based approaches and waveform-based approaches.  Waveform-based approaches, also known as Time domain approaches, use the raw waveform itself. By using these approaches, we can expect that it's relatively insensitive to hyperparameters compared to using STFT, and it's more robust to TF solution. Various model using raw waveform as input was invented, but there were largely two mainstreams: TasNet, amd Wave-U-Net.

#### 2.2.1. Time-domain Audio Separation Network (TasNet)

![TasNet](/images/TasNet.png)

**[Figure. 2]** Block diagram of TasNet.

TasNet [25] is a model mainly used for speech separation tasks, but it also achieved high performance in MSS and is still the well-known time-domain model that motivates many models. It consists of three stages: Encoder-Separation-Decoder. According to the paper that proposed it, the mixture waveform is expressed as a nonnegative weighted sum of basis signals. Let mixture signal as $\textbf{x}$, basis signal as $\textbf{B}$, and nonnegative weight as $\textbf{w}$, then $\textbf{x}$ is represented as below.

$\textbf{x} = \textbf{wB}$

When the signal of source $i$ is $\textbf{s}$<sub>$i$</sub> and masked nonnegative weight for source $i$ is $\textbf{d}$<sub>$i$</sub>$=$  $\textbf{m}$<sub>$i$</sub>$\bigodot \textbf{w}$, then $\textbf{s}$<sub>$i$</sub>$=\textbf{d}$<sub>$i$</sub>$\textbf{B}$ is satisfied. Therefore we can obtain separated sources.

For TasNet, nonnegative weights $\textbf{w}$ of $\textbf{x}$ are estimated in the Encoder composed of 1-D gated convolutional layers. Next, in the separation network, a mask $\textbf{m}$<sub>$i$</sub> is estimated through an LSTM network including skip-connection and an FC layer from the estimated $\textbf{w}$. By multiplying this with mixture weights $\textbf{w}$, weights $\textbf{d}$<sub>$i$</sub> corresponding to each source can be obtained. Finally, by multiplying with a learnable filter of Decoder that works as the basis signal $\textbf{B}$, sources can be extracted by following $\textbf{s}$<sub>$i$</sub>$=\textbf{d}$<sub>$i$</sub>$\textbf{B}$. 

Note that since the output of Encoder can also be interpreted as estimating each component $\textbf{w}$ of basis signal, it can also be interpreted as an adaptive STFT. From this perspective, TasNet can also be seen as an adaptive spectrogram masking method. Previous TF domain approaches had problems with processing inputs shorter than STFT’s window length because they used non-optimal fixed transforms. However, since TasNet does not use STFT, it has low-latency property that can handle short segments length and has the advantage of being able to separate sources in real-time. 

![conv_Tas_1](/images/conv_Tas_1.png)

**[Figure. 3]** Block diagram of Conv-TasNet.

To overcome LSTM’s long temporal dependencies problem, there is an improved model called Conv-TasNet [26], which is improved by using stacked 1-D dilated convolutional blocks including skip-connection. They achieved a better performance with a smaller model size and lower computational complexity. By doing so, it can achieve the low-power and low-latency MSS. However, it has the problem that artifacts are relatively severe when using relatively short segments.

#### 2.3.2. Wave-U-Net

![wave-u-net](/images/wave-u-net.png)

**[Figure. 4]** Block diagram of Wave-U-Net.

In general, audio signal is obtained with a high sampling rate, so it is difficult to use long temporal input due to the memory issue. However, on the contrary, in order to have a good separation performance, long-range temporal correlations are required. Another time-domain approach that attempted to solve this problem was Wave-U-Net [27]. This is a model that uses U-net architecture, which has shown good performance in computer vision area, for 1D signals and becomes one of the two mainstream time-domain models along with TasNet.

Wave-U-Net has several advantages.  First, it can obtain multi-scale features because it can create high-level features by down-sampling and combine them with high-resolution features by up-sampling due to the structure of the U-net. Second, since it reduces time resolution by half after down-sampling, it can save memory. Third, due to the skip connections, initial phase information can be preserved by delivering to the deeper layers. Two signals whose only difference is a shift in the initial phase are perceptually the same, but can have arbitrarily high direct reconstruction losses on waveforms. So it makes generative models to confuse. But by having skip connections these problems can be mitigated.[28] It will be treated more deeper in Proposed Work section.

The author also conducted various attempts to reduce border artifacts. The first is not to zero pad before performing convolution. The second is to use a technique that only interpolates between known neighboring values and keeps the first and last values not changing. Finally, linear interpolation was used instead of transposed convolutions for up-sampling to secure temporal continuity.

![Demucs](/images/Demucs.png)

**[Figure. 5]** Configuration of Demucs.

![Demucs_2](/images/Demucs_2.png)

**[Figure. 6]** Configuration of Encoder block in Demucs.

After that, there were various models motivated by Wave-U-Net and among them, the most notable model was Demucs [28]. Demucs has a U-net structure with a bidirectional LSTM inserted at the bottleneck. This is a model that targets MSS tasks in earnest, while most previous approaches were aimed at speech separation. Demucs aims to recover each stem from mixture signals unlike cocktail party problem, which means synthesis method rather than masking method. The goal of this paper is to extract four stems: (1) Drums, (2) Bass, (3) Other, and (4) Vocals from the mixture music. These are the same tasks as what we're about to do in this project. This paper also has the contribution of applying Conv-TasNet, which was created for monophonic speech separation, to MSS task. 

If the entire track is inputted, the volume of quiet and loud parts changes between the training and evaluation stages due to Layer Normalization. Therefore, it was applied by splitting the entire track into chunk units and inputting it. According to this paper, audio generated by Conv-TasNet which uses the mask learning method has less contamination from Other sources than Demucs but is vulnerable to artifacts such as constant broadband noise, hollow instrument attacks or even missing parts. The author said this is why they chose Wave-U-Net’s synthesis method rather than the masking method to solve this problem.

First, the Encoder has a largely strided convolution layer to down-sample the input features. It also has a 1×1 conv layer to increase the channel size, and by using gated linear units (GLU) decrease the channel size back. According to the paper, the combination of skip U-net connection and GLU also produces a kind of excellent (more expressive than Conv-TasNet) masking effect and is known to greatly improve performance. Next, bidirectional LSTM is inserted at the bottleneck of U-net to capture the long-range context and then enters Decoder by increasing channel size. The Decoder has a symmetric structure with the Encoder. Unlike Wave-U-Net, Transposed Convolution was used for up-sampling. By doing so, it can produce high performance by increasing channel size with lower memory and lower computational cost. According to the paper, it's important to use a high channel size to improve the performance. This property is also important to our project.

This paper also has contributions in data augmentation methods. In particular pitch/tempo shift augmentation was proven to be effective for models with many parameters such as Demucs. Detailed data augmentation will be explained in the Experiments Section.

Comparing Demucs and Conv-TasNet shows different performances for each source. Demucs performed better than Conv-TanNet on bass and drums but did not perform well on vocals and others. In the case of Demucs it is effective for instruments such as drums that have almost no harmonicity and reduces hollow artifacts in cases such as bass with strong and emphasized attack regimes because it solves phase inconsistency problems better than Conv-TasNet [29].

In addition, there are still questions about whether Demucs effectively reduced artifacts. It is well known that both image and audio signals produce upsampling artifacts when using upsampling methods [30]. Since U-net structure also includes up-sampling, it's still not free from artifact problems. Moreover, there might be some data loss by down-sampling.

#### 2.3.3. Dual-Path RNN (DPRNN)

In addition to TasNet and Wave-U-Net, another time-domain approach worth noting is DPRNN [31]. Waveform sequences generally have the characteristic of being much longer than other data, making them difficult to learn. The existing Wave-U-Net method had to learn this huge length sequence using 1D convolution with a relatively small receptive field, and the TasNet method had to rely on the filter size of the front-end encoder for learning. DPRNN was proposed to solve these problems. DPRNN consists of three stages: Segmentation, Block processing, and Overlap-Add.

![DPRNN_1](/images/DPRNN_1.png)

**[Figure. 7]** Block diagram of Segmentation stage in DPRNN.

In the Segmentation stage, the input sequence is divided into chunks of the same length K, and zero-padded appropriately so that each sample is included in all chunks the same number of times and then concatenated into a 3D tensor.

![DPRNN_2](/images/DPRNN_2.png)

**[Figure. 8]** Block diagram of Block processing stage in DPRNN.

In the Block processing stage, it is implemented as a stack of DPRNN blocks consisting of two sub-modules as above. It contains the intra- and inter-chunk operations, and each is performed iteratively and alternately. Firstly, intra-chunk RNN passes bi-directional RNN through the chunk dimension and passes the FC layer through the chunk dimension to maintain the same tensor size. Then it passes through Layer normalization to increase generalization performance and completes by giving residual connection. Inter-chunk RNN passes RNN through the time dimension. Since the tensor contains information about all chunks after passing through the intra-chunk RNN, fully sequence-level modeling is possible. 

![DPRNN_3](/images/DPRNN_3.png)

**[Figure. 9]** Block diagram of Overlap-Add stage in DPRNN.

Finally, the tensor is converted back to its original sequence form using the Overlap-Add method.

DPRNN has the advantage of being able to fully utilize global information of very long sequences with small model sizes. In addition, this DPRNN structure can also be applied to other existing Separators such as TasNet. And this dual-path sequential analysis will motivate to Band-Split RNN, which is the baseline of our projects.

### 2.4. Hybrid domain Approaches

Hybrid domain approaches usually refers to an approach that uses both Time-Frequency domain data, STFT and Time domain data, waveform as input.

#### 2.4.1. Summary of Strength & Artifacts

![Comparison](/images/Comparison.png)

**[Table. 1]** Summary of strengths and artifacts in each approaches.

Table 1 is the summary of strengths and artifacts in each approach: Spectrogram-based approaches and Time-domain approaches. Each characteristic is cited from [26] and [28]. 

There are also Solutions to each approach. In TF domain approach cases, there is Band-Split RNN as a solution and in Time domain approach cases, there are Hybrid domain approaches that use both spectrogram and waveform as input. I will discuss Band-Split RNN in the Baseline Section. In this section, I'm about to discuss on Hybrid domain approach, especially Hybrid Demucs(HDemucs)

#### 2.4.2. Hybrid Demucs(HDemucs)

As mentioned earlier, spectrogram-based methods and waveform-based methods have different performances depending on the source. In theory, the two must have the same performance, but because the size of the dataset is finite, there is inductive bias, so it creates differences. HDemucs [29] is the first end-to-end hybrid source separation model designed in a hybrid manner based on this point so that the model can flexibly obtain the advantages of each domain. Through this, it can be expected to complement each other’s different types of artifacts or lack of information. Actually, as you can see in Figure 10, there is a notable improvement, especially in Vocals and Others sources which have better performance in spectrogram-based approaches.![SDR_D_HD](/images/SDR_D_HD.png)

**[Figure. 10]** SDR comparison in Vocals: Demucs, Hybrid Demucs

![HDemucs](/images/HDemucs.png)

**[Figure. 11]** Configuration of HDemucs

![HDemucs_2](/images/HDemucs_2.png)

**[Figure. 12]** Configuration of Encoder Block.

But as mentioned earlier, the TF domain method which is split into magnitude and phase induces various problems. Among various attempts to solve these problems, the method chosen in this paper is Complex as Channels (CaC) [32]. This refers to a method of redefining STFT by performing channel-wise concatenation by dividing STFT’s real part and imaginary part. The author explains that by doing so, both magnitude and phase information can be handled by the network.

As you can see in Figure 11, HDemucs consists of two branchs: spectral branches and temporal branches. It is designed to share layers with each other at the bottleneck. The encoder of the temporal branch uses Gaussian Error Linear Units (GELU) instead of ReLU and is designed to reduce time step by 4 times for each layer. In addition, unlike Demucs, a compressed residual branch was used between convolutional layers and local attention, and biLSTM for a limited span was added to the 5th and 6th layers.

The spectral branch used CaC STFT as input and set padding size, FFT size, and hop length appropriately to make sure the time step size is the same as the final output of the temporal branch Encoder and the final output of the spectral branch encoder becomes 1 frequency size. Note that by these constraints, HDemucs has a severe problem in that it doesn't allow to change hyperparameter of STFT freely. The structure of spectral Encoder is the same as temporal Encoder but it is made to reduce frequency dimension. However, music signal is variant to translation along frequency axis. Therefore, this problem was solved by embedding frequency bin before passing through second encoder layer.

At the final bottleneck, the two branch outputs are tensors with the same size so they can be simply summed to properly utilize both information at the same time. The Decoder is symmetrically designed with Encoder and U-net skip connections. After converting the spectral branch output back to the waveform with Inverse STFT, it is summed with the temporal branch output to produce the final output.

In summary, HDemucs has two advantages. First, complement the weaknesses of each branch. Second, by the hybrid effect, better performance in Others and Vocals sources, which were not performing well in time-domain-only approaches, was obtained. On the other hand, there are also two disadvantages. First, there are lots of constraints on the STFT hyperparameter because the output shape of each branch must be the same. Second, simple summation doesn't guarantee the optimal bottleneck process. To overcome these limitations, there is the third upgraded version of Demucs, Hybrid Transformer Demucs, which is the baseline model in our project. And it will be discussed in the Baseline Section.



## 3. Baseline

### 3.1. Band-Split RNN(BSRNN)

![BSRNN](/images/BSRNN.png)

**[Figure. 13]** Block diagram of Band-Split RNN.

As briefly introduced in Table 1 as a solution of Time-Frequency domain approaches, Band-Split RNN [7] is the model that suggests a new approach to the Spectrogram-based model. Unlike the recent models, BSRNN uses only an STFT as input. It has a kind of Dual-path RNN [31] like structure that performs RNN modeling at the band-level and sequence-level. Here, the band means that it divides the STFT of the mixture into subbands according to pre-determined frequency bands.

This method, such as modeling different frequency bands independently and then segregating them, is optimized for high sampling rate signals. Moreover, since each instrument has different spectral features, by determining appropriate band split rules depending on sources, it can secure source adaptive frequency resolution and it is optimized for MSS task.

As depicted in Figure 13, BSRNN is divided into three submodules. The first is the Band Split Module which is the main idea of this model. It splits an STFT with frame size $T$ into $K$ subbands according to pre-determined subbands and then processes each with Layer Normalization and FC layer to unify all different lengths of frequency to the same size $N$. Here the input STFT uses CaC spectrogram as seen in HDemucs, so that complex-valued spectrogram is converted to a real-valued subband. Note that each Normalization or FC layer does not share with each other so it has the characteristic of being able to handle different frequency bands independently. Each of these $K$ subband features has a size of $N \times T$ so it can be stacked into a 3D tensor of $N \times K \times T$.

Next is the Band and Sequence Modeling Module. This module was inspired by DPRNN’s Dual-Path method. Unlike DPRNN, which processes intra-chunk and inter-chunk, RNN modeling is performed through sequence-level RNN which has $T$ dimension, and band-level RNN which has $K$ dimension respectively. Each RNN module is implemented similarly to DPRNN as seen in the figure above.

In the end, it passes through the Mask Estimation Module which estimates complex-valued TF mask for the target source. It is implemented by passing MLP after unfolding the 3D tensor into the 2D tensor again. The reason for using MLP here is that MLP is known to be more effective than a single FC layer in mask estimation [33].

The author said that in general speech signal processing, these kinds of Group-splitting modules did not work well because they don't have their own frequency-dependent patterns, but in MSS different instruments have different frequency characteristics so that the frequency band split method helps. The optimal band split rules for Vocals suggested in this paper is like below.

![BSRNN_rule](/images/BSRNN_rule.png)

**[Table. 2]** Band split rules for Vocals source suggested in the BSRNN paper.

### 3.2. Hybrid Transformer Demucs(HTDemucs)

![HTDemucs](/images/HTDemucs.png)

**[Figure. 14]** Configuration of HTDemucs.

![HTDemucs_2](/images/HTDemucs_2.png)

**[Figure. 15]** Configuration of Cross-Domain Transformer Encoder in HTDemucs(Left) and Transformer Encoder layer(Right)

HTDemucs [6] uses Encoder and Decoder structures as they are in HDemucs and even the same configuration, but replaces the innermost Encoder-Decoder which consists of local attention and bi-LSTM with cross-domain Transformer Encoder using self-attention for each branch and cross-attention between different branches.

In HDemucs, there was a constraint on selecting STFT’s hyperparameters in order to combine the output tensors of different branches' Encoder at the bottleneck layer as mentioned earlier. However, by using Transformer in HTDemucs, it became free from such size constraints. As mentioned later, such characteristics increase flexibility when changing the architecture of the spectral branch in this project.

The output passed through the Encoder in each branch passes through the Cross-Domain Transformer Encoder which normalizes first and then performs 1D and 2D sinusoidal encoding like other transformers. Then flatten the spectral branch tensor into a 1D tensor then pass through the Transformer Encoder block to calculate the attention. In the self-attention (Transformer) Encoder block, attention consists of 8 heads and the hidden state of the Feed-Forward (FF) network is 4 times of transformer dimension. To stabilize training, Layer Normalization is performed before attention and FF, and finally, one transformer encoder block is completed through time layer normalization. Cross-attention block is the same as self-attention but only uses a tensor from another branch when calculating attention. Generally, the transformer has severe memory consumption depending on sequence length so input waveform length was secured within limited memory by using a sparse attention kernel.

![SDR_HTD](/images/SDR_HTD.png)

**[Figure. 16]** Average SDR comparison : Demucs, Hybrid Demucs, Hybrid Transformer Demucs, Band-Split RNN

This model achieved the SOTA performance as above.



## 4. Proposed Works

### 4.1. Imbalanced branch problems in HTDemucs

In general, algorithms dealing with audio signals are mostly divided into whether they are time-domain approaches or spectral-domain approaches. Over time Hybrid-domain approach has achieved great success and the current SOTA model is Hybrid Transformer Demucs [6] which is a hybrid version of Demucs. 

The effect we expect from the hybrid approach is a more accurate and realistic separation performance due to not simply an increase in model size, but the supplementation of information from both domains. Therefore, it is necessary to check whether the current HTDemucs fulfills these expectations. As shown in the previous section, HTDemucs is divided into two branches, each responsible for a time-domain and a spectral-domain approach. Therefore, checking the output of each branch just before they are combined is sufficient to analyze the effectiveness of the hybrid.

![spec_HTD](/images/spec_HTD.PNG)

**[Figure. 17]** Spectrogram Comparison in the output of HTDemucs and each branch

*(Audio is available in \audios\mixture.wav)*

**[Audio. 1]** Mixture sound (Be careful for LOUD sound)

*(Audio is available in \audios\gt.wav)*

**[Audio. 2]** Ground Truth Vocal sound

*(Audio is available in \audios\HTD_hybrid.wav)*

**[Audio. 3]** HTDemucs output sound

*(Audio is available in \audios\HTD_spectral.wav)*

**[Audio. 4]** Spectral branch output in HTDemucs

*(Audio is available in \audios\HTD_time.wav)*

**[Audio. 5]** Temporal branch output in HTDemucs

The figure above shows the spectrogram of the final output and the spectrogram of the output of each branch. As you can see, the time branch does not give any meaningful results(almost silence) and the frequency branch is doing most of the work. Since Demucs which is the original version of HTDemucs performs well as a waveform-domain model, it is difficult to interpret that the time branch is designed incorrectly. Therefore, we can infer that the problem is that the wrong design of the frequency branch is preventing the desired hybrid effect. At the very least, it's reasonable to think that when a model that started out as a good time-domain model is in trouble, then the frequency branch should be the one to fix it. We don't have a good reason why a bad frequency branch design would cause only the frequency branch to perform well, but we hypothesized that a bad design of one branch would cause both branches to be unable to play a balanced role. Another reason to think these ways is the design of the frequency branch. The spectral branch is a simple copy of the counterpart structure and there is a lack of attempts to properly process spectrogram compared to temporal branch. 

On the other hand, Band Split RNN [7] which achieved best performance among recent spectral-domain algorithms is optimized for MSS task. In fact, in the paper that proposed HDemucs [29], they mentioned that they had experimented with a band adaptive approach by giving different weights to different frequency bands, and they observed it was not beneficial. But we felt that this was not sufficient. It is worthwhile to experiment with replacing spectral branch in HTDemucs with Band Split RNN method and developing an algorithm that can work with.

We can also look at these proposals from a different perspective. BSRNN went back to the old TF-domain approach unlike the recent trends. However as mentioned repeatedly, spectrogram-based and waveform-based methods have different artifacts and strengths. HDemucs showed noticeable performance improvement in Others and Vocals which were weak in existing waveform-domain approach (Demucs) by showing that Hybrid method complements each other’s weaknesses of different approaches. Therefore, it can be seen as an attempt to solve BSRNN’s problems by supplementing BSRNN with hybrid-domain approach.

### 4.2. Proposal

Before implementing our proposals, it is necessary to review whether the model we chose in each branch is an optimal choice. Even though the current problem in the time branch is that artifacts can occur during the up-sampling process, the U-net structure is not avoidable in the waveform-based method in order to avoid running out of memory problems. Therefore, choosing Demucs for the time branch makes sense. As mentioned in BSRNN, the spectral branch splits the frequencies into predetermined subbands and performs independent modeling for each of them, which helps to effectively model the music signals that have different frequency components for each source.

However, before combining BSRNN and HTDemucs, there are a few inconsistencies between them. While BSRNN is a task that separates a single target source, HTDemucs separates multi-target sources. To maintain BSRNN’s philosophy, we changed HTDemucs to a single target source separation model by reducing the output channel of the last Decoder to 2, which is the stereo channel size. After sufficiently learning one target source, we can fine-tune using the transfer learning method for other sources. We chose the target source as Vocals because we can expect the biggest performance changes than other sources due to the improvement of the spectral branch. This is why our project title is Vocal Separation. Including this modification, we proposed our model architecture as below.

![Proposed](/images/Proposed.png)

**[Figure. 18]** Configuration of our proposed model.

As you can see, the lower branch is the Temporal branch, which is the same as HTDemucs one except for the output channel size, and the upper branch is the Spectral branch, which is similar to the Band-Split RNN structure. BSRNN consists of three submodules: Band Split Module, Band and Sequence Modeling module, and Mask Estimation Module as mentioned earlier. In order to combine two branches at the bottleneck in line with HTDemucs’ philosophy, we need to consider how to combine the outputs of the two branches. First, we can consider the Band Split Module and Band and Sequence Modeling module as Encoder because they provide high-level features from input data. Also, we can consider the Mask Estimation Module as a Decoder because it makes an output from high-level features and the structure is almost symmetric with the Encoder. Therefore, as you can see in Figure 18, after forwarding the Band and Sequence Modeling Module, they are combined with temporal encoded features in the Cross-Domain Transformer Encoder.

There are several more modifications in Band-Split RNN to combine each model.

![attention](/images/attention.png)

**[Figure. 19]** Attention mechanism in Cross Transformer based on the choice of axis in spectral branch.

First, before computing attention, we had to choose which axis to consider as token size and which axis to consider as hidden size. For the time branch, as HTDemucs did, it’s obvious that $C$ is the hidden size and $T$ is the token size, and that's what we did. For the frequency branch, we need to choose one of $K$ and $N$ to be the hidden size and the other to be flattened with $T$ to be the token size. ($K$: Number of split subbands, $N$: Dimension of converted bandwidth by FC)

If we choose $N$ as the hidden size, the Transformer Encoder layer will calculate the self-attention between different ($T$,$K$) pairs(tokens).(Upper in Figure 19) However, this is a repetition of the same modeling because the band-sequence modeling module has already modeled the correlation between each other. On top of that, it’s hard to expect a good analysis when modeling the correlation between different subbands at different time steps. In the Cross-attention Encoder cases, the layer calculates the cross-attention with the $T$th time step of the token which is the result of the self-attention of the time branch. Therefore, if there are not enough attention heads, the token in the time branch can only learn a limited number of ways to calculate the correlation with tokens in the frequency branch's many time steps and many subbands. If the key to the success in HTDemucs was the attention to the overall frequency component within a single time step, this method can be fatal as it forces the model to pay attention to only a few subbands. From the frequency branch's point of view, it is difficult to get meaningful results because the correlation with the time branch's many time steps is calculated with the limited data of one subband. Therefore, it is difficult to achieve the desired effect of cross-domain attention.

If $K$ is chosen as the hidden size, the Transformer Encoder layer computes the self-attention between different ($T$,$N$) pairs(tokens).(Lower in Figure 19) In other words, it models the correlation between a token created by a combination of different extracted features for each subband at a given time step and a token created by a combination of various features at another timestep. It is difficult to expect a good effect between tokens in the same time step, but if the tokens are in different time steps, it’s possible to calculate the attention in various way from the same time step pairs due to the various tokens from one frequency bin. In the Cross-attention Encoder layer cases, calculates the cross-attention with the token at the $T$th time step after passing the self-attention of the time branch. This time, even if there are fewer attention heads, the token at one time step from the time branch will calculate attention with the token in the frequency branch containing all subband features.

![SDR_NK](/images/SDR_NK.png)

**[Figure. 20]** SDR results based on choice of axis in spectral branch. Blue line is the result of  HTDemcus, the baseline model. Green line is when we choose $K$ as the hidden size. And red, purple, and brown line is when we choose $N$ as the hidden size with different bandwidth rule V2,V3, and V4(suggested in BSRNN paper). Orange line is the result of BSRNN.

As you can see in Figure 20, it's obvious that it has superior performance when we choose $K$ as the hidden size(Green). For these reasons, we chose $K$ as the hidden size and the token size to be $N \times T$ by flattening. Although we approached $K$ axis as a temporal sequence by performing RNN across $K$ in the band-sequence module, this does not mean that it should not be viewed as a channel. Rather, these modelings allow us to view the tensor in different ways, both temporally and as a channel.

The next modification is the band split rule. As mentioned earlier, HTDemucs is free from the constraint of the tensor size in each branch by using a cross-domain Transformer Encoder. Therefore, there are no constraints on the size of $N$,$T$ in BSRNN’s output tensor. However, the channels, the hidden size in the Transformer, of the two branches must be the same. In addition, in order to use 2D sinusoidal positional encoding before inputting to transformer, input channel must always be a multiple of 4.

Therefore, K and C must be agreed to be multiples of 4 and we propose two for this. The first is $K=C=40$, which we matched with the subband rule proposed for vocals & others in the BSRNN paper. The difference is that the highest frequency bands were combined into one subband to ensure a multiple of 4. The second is $K=C=384$ matched with Demucs’ output channel. In the former case, memory can be saved and the model size is relatively small. However, Demucs’ output channel decreased from 384 to 40 by about 10%. Considering that using high channels greatly improves performance as explained in Demucs paper, this makes it difficult to expect good effects. The latter keeps the output channel of Demucs, but the memory overhead becomes too large if $N$ is the same as BSRNN paper. However, if we think about the meaning of $K$, increasing $K$ means that the subbands are cut more tightly. Therefore, we can expect to lose relatively little information even though decreasing $N$, because the subband bandwidth is not large. Also, if $N$ is too large, the number of tokens which is ($N$,$T$) becomes too large, so it is unfavorable for computing attention. For this project, we reduced $N$ from 128 to 22 (the most we could do within the limits of our GPU memory).

![Proposed_rule](/images/Proposed_rule.png)

**[Table. 3]** Band split rule in our proposed model.

The third modification is shrinking the model size. There are many FC layers because we set the $K$ as a large number. So we must shrink the model size to overcome these overheads. We reduce the Cross Transformer Encoder layers 5 to 3 and reduce the Dual-path RNN layers 12 to 8. By doing so, we obtained the model size of our proposal 85.9MB, which is the smallest compared to other Baseline models. This is an important fact because the reason we got better performance than others is not by the model size but improvement in the Hybrid effect.

![modelsize](/images/modelsize.png)

**[Table. 4]** The model size comparison

The last modification we made was in Objective function. HTDemucs uses direct reconstruction losses on waveforms as an objective function. It computes the L1 loss between estimated waveform and waveform for each time step. In fact, the paper that proposed Demucs said that there is no notable difference between L1 loss and L2 loss, and we used these losses in this project. However, we added another loss term to compensate for the phase-inconsistency. In the case of HTDemucs, the intital phase is sufficiently delivered due to the existence of skip U-net connections, but this advantage is lost by switching to BSRNN. To solve this problem, we added the L1 loss for the real and imaginary parts of the spectrogram. Using loss for the spectrogram can alleviate initial phase inconsistency problem [34], so we chose the loss as follows: 

![objective](/images/objective.png)



## 5. Experiments

### 5.1. Setup

#### 5.1.1. Dataset

The currently most widely used dataset for music source separation tasks is MUSDB18(HQ). [4,5] It consists of 150 songs made by mixing 4 stems: drums, bass, vocals, and others. We used 85 of them as the training set, 15 as the validation set, and 50 as the test set. 

Each song is about 3 to 5 minutes long, so we cut it into segments before inputting it into the model. For this project, we chose 4 seconds. On the other hand, the paper [6] that proposed HTDemucs showed that increasing the input segment length can significantly improve the performance by being able to capture long temporal range contextual information. However, to perform the remix augmentation mentioned later, the batch size must be at least 2, so we used 4 seconds, which is relatively shorter than HTDemucs used. However, this is not too short considering that BSRNN used 3 seconds as an input segment.

For the STFT, which is the input to the frequency branch, we set the DFT frequency size, n_fft, to 4096, and the hop_length to 1024, which is 1/4 of n_fft, the same as HTDemucs used.

#### 4.1.2. Data Augmentations

To train Transformer well, we need a lot of data, and the MUSDB18 dataset is not enough. To solve this problem, we used several data augmentations also used in HTDemucs.

1) Pitch/Tempo shift: shifts the pitch of the audio to a random semitone between -2 and +2, and randomly stretches the audio along the time axis. 
2) Random temporal shift: Obtains a random offset between 0 and 1 seconds, and then cuts off the consecutive 4 seconds from the input segments to prevent the same segment from entering every epoch.
3) Audio Scaling: Scales the value of the audio tensor by a random value between 0.25 and 1.25.
4) Flip channel/sign: Randomly toggles the sign of the audio tensor's value between + and -, or randomly toggles the 2 channels of stereo audio.
5) Remix: This augmentation creates a new song by randomly mixing the stems from multiple songs in a given batch.

#### 5.1.3. Training

We used the optimizer as ADAM optimizer, momentum(beta1) as 0.9, and beta2 as 0.999. And the learning rate is selected as 3e-4. All these settings are the same as that of HTDemucs. The batch size we select was 2, so the remix augmentation will be performed with 2 different songs.

Also we trained BSRNN 50 epochs due to the poor results at the end and trained other models 100 epochs. 

#### 5.1.4. Metric

The most widely used metric in the music source separation task using MUSDB is the Signal-to-Distortion Ratio (SDR) [35]. It is defined as follows:

![SDR_formula](/images/SDR_formula.PNG)

However, for simplicity and fast evaluation, new SDR (nSDR) was proposed in HDemucs paper [29] and is defined as follows:

![nSDR_formula](/images/nSDR_formula.PNG)

In our project, we used nSDR for validation.



### 5.2. Results

#### 5.2.1. Loss

![lossplot](/images/lossplot.png)

**[Figure. 21]** The loss plot of proposed model

As depicted in Figure 21, we obtained training loss and validation loss over epochs, and they all monotonically decreased. By this fact, we can infer that training was stable, and there is no overfitting.

#### 5.2.2. SDR

![SDR_proposed_2](/images/SDR_proposed_2.png)

**[Table. 5]** The best SDR comparison

As shown in Table. 5, we can see that the SDR score is significantly improved compared to the baseline model. The reason why it is different from the SDR of the baseline model reported is due to the difference in the experimental setting. In fact, in the paper which proposed the HTDemucs, they used the segment length of 7.8 seconds. The paper claims that the segment length plays an important role and that increasing 4 seconds increased the SDR by nearly 0.7㏈. However, as mentioned earlier, we only used 4 seconds due to memory issues. Also, since the transformer requires a large dataset, the author who proposed HTDemucs used an additional 800 songs in addition to the MUSDB18(HQ) dataset. Moreover, original HTDemucs used a batch size of 64 and performed remix augmentation in groups of 4 songs, so a richer dataset could be obtained. In the case of BSRNN, the original proposed method was designed to discard segments containing a lot of silence (more than 50%) when performing augmentation, so the actual reported results may be different. Considering these points, the SDR of the SOTA model may have been lower than that reported. However, in our experiments, these poor settings were the same for all models, so we can conclude that our proposed model is an improvement over the baseline model. Furthermore, if we had performed the same augmentation proposed in the baseline paper using sufficient GPU memory, the proposed model could have achieved much better results.

#### 

![SDR_proposed](/images/SDR_proposed.png)

**[Figure. 22]** The SDR comparison

Figure 22 shows the SDR results in every epoch. The reason that HTDemucs show relatively early saturation is that as mentioned earlier, the time branch becomes almost silent, so after some epochs, the model changed to one branch architecture, so the curve was bent. On the other hand, our model shows balanced performance in each branch as depicted later, so the curve is steadily increased without bending. The reason that BSRNN is getting worse after about 30 epochs is due to the ADAM optimizer might be felled to the wrong optimum to our best knowledge.

Based on the above conclusions, it is significant that the proposed model achieves higher SDR with a smaller model size than the other two baseline models, as mentioned in section 4.2. The possible reason for this is that by reducing the dimension of FC layer $N$ to a much smaller size instead of increasing the number of subbands much more than the baseline method, the model size of all three submodules of BSRNN can be significantly reduced while securing high-frequency resolution. In addition, by using band-sequence dual path RNNs, we can expect sufficient effects even if we use less attention layer. In the case of HTDemucs, instead of multiple encoder & decoder layers including convolutional layers with high channels, one FC layer of 22 channels per subband is replaced, which is a significant gain in model size.

#### 5.2.3. Spectrograms & Samples

![spec_comparison](/images/spec_comparison.PNG)

**[Figure. 23]** The Spectrogram comparison

*(Audio is available in \audios\mixture.wav)*

**[Audio. 1]** Mixture sound (Be careful for LOUD sound)

*(Audio is available in \audios\gt.wav)*

**[Audio. 2]** Ground Truth Vocal sound

*(Audio is available in \audios\HTD_hybrid.wav.wav)*

**[Audio. 3]** HTDemucs output sound

*(Audio is available in \audios\BSRNN.wav)*

**[Audio. 6]** BSRNN output sound

*(Audio is available in \audios\Proposed_hybrid.wav)*

**[Audio. 7]** Ours output sound

The above figures show the summary of the baseline models and the proposal models' performances. As you can see from the spectrogram, our model captures the high-frequency components better than BSRNN due to the time branch. Our model also captures the low-frequency components better than HTDemucs due to the improvement in the spectral branch. These will be discussed in detail in Figure 24. When we hear the sample, our model has fewer other source components compared to HTDemucs. And our model has less hollow sounds compared to BSRNN.

![spec_proposed_branch](/images/spec_proposed_branch.PNG)

**[Figure. 24]** The Spectrogram comparison in each branch

*(Audio is available in \audios\Proposed_spectral.wav)*

**[Audio. 8]** Spectral branch output in the Proposed model

*(Audio is available in \audios\Proposed_time.wav)*

**[Audio. 9]** Temporal branch output in the Proposed model

The figure above shows a spectrogram of the output of each branch to check whether we have solved the imbalance performance problem of each branch of HTDemucs. As shown in 'Ours : Time branch', we can see that it contains much more meaningful results compared to the time branch output of the previous HTDemucs. In particular, we can see that the Temporal branch is trained to capture high frequency components, while the Spectral branch is trained to capture low frequency components. The reason for this is that in the case of audio signals, most of the energy is concentrated in low frequencies, so branches with spectrograms as input are stronger for low frequency analysis, and in the case of waveforms, they are relatively stronger for high frequency analysis because they can capture transient events such as sharp edges and fast variations that are difficult to capture in STFT. Considering the branch imbalance problem in HTDemucs, it proves that by fixing the frequency branch, the time branch can also play an even role. As a result, as depicted in Figure 22, we achieved monotonically increasing performance, which HTDemucs couldn't achieve due to the imbalance branch performance. 



## 6. Conclusions & Discussions

In our work, we found the problems in the current hybrid models, which is the imbalance branch problem, starting from the simple question that why Hybrid approaches work. We solve these problems by improving spectral analysis in Hybrid Transformer Demucs inspired by Band-Split RNN. When we tried to combine them, we encountered several problems in implementations such as Initial Phase Inconsistency, Selection of hidden size, and Memory problems. To solve these, we tried various attempts by comparing the results and also we used the appropriate objective function.

Our approaches have two contributions: First, we improved the spectral branch in Hybrid Transformer Demucs to maximize the hybrid effect. Second, we adapted the hybrid approach to Band-split RNN.

We achieved better performance than two baseline models with smaller model size. It’s because by using dense subbands with low FC dimension, we can reduce the model size maintaining high frequency resolution. And we also observed the balanced hybrid performance in each branch by analyzing the spectrogram. By these results, we prove that good design in one branch can affect other side branch in Cross Transformer. Because we didn’t modify time branch at all, but time branch performance was significantly improved. Furthermore, we observed the hybrid approach effect by adapting hybrid approach to BSRNN.



## Reference

[1] E. Colin Cherry. Some experiments on the recognition of speech, with one and with two ears. The Journal of the Acoustic Society of America, 1953.

[2] S. R. Livingstone, K. Peck, and F. A. Russo, “Acoustic differences in the speaking and singing voice,” in Proceedings of Meetings on Acoustics ICA2013, vol. 19, no. 1. Acoustical Society of America, 2013, p. 035080.

[3] Nobutaka Ono, Zafar Rafii, Daichi Kitamura, Nobutaka Ito, and Antoine Liutkus, “The 2015 Signal Separation Evaluation Campaign,” in International Conference on Latent Variable Analysis and Signal Separation(LVA/ICA), Aug. 2015.

[4] Zafar Rafii, Antoine Liutkus, Fabian-Robert Stoter, Stylianos Ioannis Mimilakis, and Rachel Bittner, "The musdb18 corpus for music separation", 2017.

[5] Zafar Rafii, Antoine Liutkus, Fabian-Robert Stoter, Stylianos Ioannis Mimilakis, and Rachel Bittner, “Musdb18-hq - an uncompressed version of musdb18", Aug. 2019.

[6] Simon Rouard, Francisco Massa, Alexandre Défossez, "Hybrid transformers for music source separation", 2022 arXiv:2211.08553.

[7] Yi Luo, Jianwei Yu, "Music source separation with Band-split RNN", 2022        arXiv:2209.15174.

[8] E. M. Grais and H. Erdogan, "Single channel speech music separation using nonnegative matrix factorization and spectral masks," 2011 17th International Conference on Digital Signal Processing (DSP), Corfu, Greece, 2011, pp. 1-6, doi: 10.1109/ICDSP.2011.6004924.

[9] K. W. Wilson, B. Raj, P. Smaragdis and A. Divakaran, "Speech denoising using nonnegative matrix factorization with priors," 2008 IEEE International Conference on Acoustics, Speech and Signal Processing, Las Vegas, NV, USA, 2008, pp. 4029-4032, doi: 10.1109/ICASSP.2008.4518538.

[10] P. -S. Huang, S. D. Chen, P. Smaragdis and M. Hasegawa-Johnson, "Singing-voice separation from monaural recordings using robust principal component analysis," 2012 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Kyoto, Japan, 2012, pp. 57-60, doi: 10.1109/ICASSP.2012.6287816.

[11] N. Roman and J. Woodruff, "Ideal binary masking in reverberation," 2012 Proceedings of the 20th European Signal Processing Conference (EUSIPCO), Bucharest, Romania, 2012, pp. 629-633.

[12] Yipeng Li, DeLiang Wang, "On the optimality of ideal binary time–frequency masks", Speech Communication, Volume 51, Issue 3, 2009, pp. 230-239, ISSN 0167-6393

[13] Shasha Xia, Hao Li, Xueliang Zhang, "Using Optimal Ratio Mask as Training Target for Supervised Speech Separation", 2017,         arXiv:1709.00917

[14] P. -S. Huang, M. Kim, M. Hasegawa-Johnson and P. Smaragdis, "Deep learning for monaural speech separation," 2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Florence, Italy, 2014, pp. 1562-1566, doi: 10.1109/ICASSP.2014.6853860.

[15] Huang, Po-Sen, Minje Kim, Mark A. Hasegawa-Johnson and Paris Smaragdis. “Singing-Voice Separation from Monaural Recordings using Deep Recurrent Neural Networks.” International Society for Music Information Retrieval Conference (2014).

[16] Chandna, Pritish, Marius Miron, Jordi Janer and Emilia Gómez. “Monoaural Audio Source Separation Using Deep Convolutional Neural Networks.” Latent Variable Analysis and Signal Separation (2017).

[17] A. Narayanan and D. Wang, "Ideal ratio mask estimation using deep neural networks for robust speech recognition," 2013 IEEE International Conference on Acoustics, Speech and Signal Processing, Vancouver, BC, Canada, 2013, pp. 7092-7096, doi: 10.1109/ICASSP.2013.6639038.

[18] Andrew J.R. Simpson, Gerard Roma, Mark D. Plumbley, "Deep Karaoke: Extracting vocals from musical mixtures using a convolutional deep neural network", 2015, arXiv:1504.04658

[19] Jansson, Andreas, Eric J. Humphrey, Nicola Montecchio, Rachel M. Bittner, Aparna Kumar and Tillman Weyde. “Singing Voice Separation with Deep U-Net Convolutional Networks.” International Society for Music Information Retrieval Conference (2017).

[20] Z. -C. Fan, Y. -L. Lai and J. -S. R. Jang, "SVSGAN: Singing Voice Separation Via Generative Adversarial Network," 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Calgary, AB, Canada, 2018, pp. 726-730, doi: 10.1109/ICASSP.2018.8462091.

[21] D. S. Williamson, Y. Wang and D. Wang, "Complex Ratio Masking for Monaural Speech Separation," in IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 24, no. 3, pp. 483-492, March 2016, doi: 10.1109/TASLP.2015.2512042.

[22] Liang S, Liu W, Jiang W, Xue W. The optimal ratio time-frequency mask for speech separation in terms of the signal-to-noise ratio. J Acoust Soc Am. 2013 Nov;134(5):EL452-8. doi: 10.1121/1.4824632. PMID: 24181990.

[23] H. Erdogan, J. R. Hershey, S. Watanabe and J. Le Roux, "Phase-sensitive and recognition-boosted speech separation using deep recurrent neural networks," 2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), South Brisbane, QLD, Australia, 2015, pp. 708-712, doi: 10.1109/ICASSP.2015.7178061.

[24] Zhuo Chen, Yi Luo, Nima Mesgarani, "Deep attractor network for single-microphone speaker separation", 2017, arXiv:1611.08930

[25] Yi Luo, Nima Mesgarani, "TasNet: time-domain audio separation network for real-time, single-channel speech separation", 2017, arXiv:1711.00541

[26] Yi Luo, Nima Mesgarani, "Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation", 2018,  arXiv:1809.07454

[27] Daniel Stoller, Sebastian Ewert, Simon Dixon, "Wave-U-Net: A Multi-Scale Neural Network for End-to-End Audio Source Separation", 2018, arXiv:1806.03185

[28] Alexandre Défossez, Nicolas Usunier, Léon Bottou, Francis Bach. "Music Source Separation in the Waveform Domain", 2021, hal-02379796v2.

[29] Alexandre Défossez, "Hybrid spectrogram and waveform source separation", 2021, arXiv:2111.03600

[30] Jordi Pons, Santiago Pascual, Giulio Cengarle, Joan Serrà, "Upsampling artifacts in neural audio synthesis", 2020, arXiv:2010.14356

[31] Y. Luo, Z. Chen and T. Yoshioka, "Dual-Path RNN: Efficient Long Sequence Modeling for Time-Domain Single-Channel Speech Separation," ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Barcelona, Spain, 2020, pp. 46-50, doi: 10.1109/ICASSP40776.2020.9054266.

[32] Woosung Choi, Minseok Kim, Jaehwa Chung, Daewon Lee, Soonyoung Jung, "Investigating U-Nets with various Intermediate Blocks for Spectrogram-based Singing Voice Separation", 2019, arXiv:1912.02591

[33] K. Li and Y. Luo, “On the use of deep mask estimation module for neural source separation systems,” Proc. Interspeech, 2022

[34] Défossez, Alexandre, et al. "Sing: Symbol-to-instrument neural generator." Advances in neural information processing systems 31 (2018).

[35] Vincent, Emmanuel, Rémi Gribonval, and Cédric Févotte. "Performance measurement in blind audio source separation." IEEE transactions on audio, speech, and language processing 14.4 (2006): 1462-1469.
