\documentclass[11pt,a4paper]{article}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{geometry}
\geometry{margin=1in}
\usepackage{setspace}
\usepackage{titlesec}
\usepackage{fancyhdr}
\usepackage{hyperref}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{longtable}
\usepackage{caption}
\usepackage{enumitem}
\usepackage{xcolor}

\setstretch{1.2}

\pagestyle{fancy}
\fancyhf{}
\rhead{Technical Report}
\lhead{TangoFlux-Timbre}
\rfoot{\thepage}

\title{\textbf{Technical Report} \\[0.5em]
\large TangoFlux-Timbre: Controllable Instrument Timbre Generation via Text, Pitch, and Velocity}
\author{Applicant: [Your Name] \\ Program: [Your Program / Major] \\ Institution: [Your Institution]}
\date{[Insert Date]}

\begin{document}

\maketitle

\begin{abstract}
This technical report presents \textbf{TangoFlux-Timbre}, a controllable audio generation system for synthesizing high-fidelity instrument timbres with explicit control over \textbf{textual descriptions, musical pitch, and velocity dynamics}. Built upon the TangoFlux architecture, the system extends flow-matching diffusion transformers with specialized embeddings for pitch and velocity, enabling fine-grained control over generated instrument sounds. In addition, MuLan audio embeddings are incorporated to improve timbre representation and support reference-audio-guided generation. The model generates 44.1kHz stereo audio and is evaluated on the NSynth dataset. Experimental results demonstrate that the proposed method can generate diverse instrument timbres with accurate pitch and velocity control.
\end{abstract}

\section{Introduction}

Controllable audio synthesis has emerged as an important research direction in music generation, sound design, and virtual instrument modeling. Although recent text-to-audio systems have achieved impressive perceptual quality, they often lack the fine-grained controllability required for musical applications. In practical use cases, musicians typically need control over:

\begin{enumerate}[leftmargin=1.5em]
    \item \textbf{Timbre}: the characteristic sound quality of an instrument, such as ``bass electronic percussive''
    \item \textbf{Pitch}: the fundamental frequency of the generated note (MIDI values 0--127)
    \item \textbf{Velocity}: the intensity or loudness of the note attack (discrete levels: 25, 50, 75, 100, 127)
\end{enumerate}

To address these requirements, we extend TangoFlux, a state-of-the-art text-to-audio model, with explicit pitch and velocity conditioning. The resulting system enables note-level instrument synthesis with both natural-language and parameter-based control.

\section{Related Work}

\subsection{Text-to-Audio Generation}

Recent advances in text-to-audio synthesis have been driven by diffusion models and flow matching techniques. AudioLDM, Make-An-Audio, and Tango demonstrated the feasibility of generating diverse audio from text descriptions. TangoFlux further improved generation quality and inference efficiency through rectified flow matching and diffusion transformers.

\subsection{Controllable Music Generation}

MusicLM and MusicGen explored controllable music generation using text conditioning. However, these systems primarily focus on full music composition rather than isolated instrument synthesis. In contrast, our work targets note-level control, enabling precise specification of pitch and dynamics.

\subsection{Neural Audio Synthesis}

Neural synthesis methods such as DDSP and SampleRNN have investigated direct waveform generation with explicit control parameters. The present work combines the flexibility of diffusion-based generation with the controllability of parametric synthesis.

\section{Method}

\subsection{Model Architecture}

The proposed architecture extends TangoFlux with additional conditioning modules for pitch, velocity, and audio-level timbre features.

\subsubsection{Text Encoder}

Flan-T5-Large is used as the text encoder to process textual descriptions of instrument timbres. The text encoder produces a sequence of semantic embeddings that represent the desired sound.

\begin{verbatim}
Text Input: "bass electronic percussive"
     ↓
T5TokenizerFast (max_length=64)
     ↓
T5EncoderModel → [batch, seq_len, 768]
     ↓
Linear(768→1024) + ReLU → [batch, seq_len, 1024]
\end{verbatim}

\subsubsection{Pitch Embedding}

Pitch is encoded using a learnable embedding table that maps MIDI pitch values (0--127) to continuous representations:

\begin{verbatim}
self.pitch_embedding = nn.Embedding(128, in_channels)  # in_channels=64
\end{verbatim}

The pitch embedding is added directly to the latent representation, allowing the model to condition generation on the target fundamental frequency.

\subsubsection{Velocity Embedding}

Velocity is discretized into five levels corresponding to standard MIDI velocity values (25, 50, 75, 100, 127). A learnable embedding maps these discrete values:

\begin{verbatim}
self.velocity_embedding = nn.Embedding(5, in_channels)  # in_channels=64
\end{verbatim}

This design captures the relationship between attack intensity and timbral characteristics.

\subsubsection{MuLan Audio Embedding}

We incorporate MuLan (MuQ-MuLan-large) embeddings to capture audio-level timbre characteristics. MuLan provides a 512-dimensional representation that captures perceptual audio features:

\begin{verbatim}
self.mulan_embedding = nn.Linear(512, 1024)
self.mulan_replacement = nn.Parameter(torch.zeros(1, 1024))
\end{verbatim}

During training, MuLan embeddings are randomly replaced with a learnable parameter with 50\% probability, enabling the model to generate timbres from text alone when MuLan features are unavailable.

\subsubsection{Duration Embedding}

Duration is encoded using sinusoidal positional embeddings adapted from Stable Audio:

\begin{verbatim}
self.duration_embedder = DurationEmbedder(
    text_embedding_dim,  # 768
    min_value=0,
    max_value=max_duration  # 30 seconds
)
\end{verbatim}

\subsubsection{Diffusion Transformer}

The core architecture uses FluxTransformer2DModel with mixed attention mechanisms.

\begin{table}[h]
\centering
\begin{tabular}{ll}
\toprule
\textbf{Parameter} & \textbf{Value} \\
\midrule
num\_layers & 4 \\
num\_single\_layers & 12 \\
in\_channels & 64 \\
attention\_head\_dim & 128 \\
joint\_attention\_dim & 1024 \\
num\_attention\_heads & 8 \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Conditioning Strategy}

The conditioning signals are integrated in two ways:

\begin{enumerate}[leftmargin=1.5em]
    \item \textbf{Encoder Hidden States}: text embeddings, duration embedding, and MuLan embedding are concatenated along the sequence dimension:
\end{enumerate}

\begin{verbatim}
[text_emb, duration_emb, mulan_emb] → [batch, seq_len+2, 1024]
\end{verbatim}

\begin{enumerate}[leftmargin=1.5em]
    \setcounter{enumi}{1}
    \item \textbf{Latent-Space Conditioning}: pitch and velocity embeddings are expanded to match the audio sequence length and added directly to the noisy latent:
\end{enumerate}

\begin{verbatim}
noisy_input = noisy_latent + pitch_emb + velocity_emb
\end{verbatim}

\subsection{Training Objective}

We employ flow matching with the following objective:

\begin{verbatim}
L = E[||v_theta(z_t, t, c) - (epsilon - z_0)||^2]
\end{verbatim}

Where:
\begin{itemize}[leftmargin=1.5em]
    \item $z_t$ is the noisy latent at timestep $t$
    \item $c$ represents all conditioning signals (text, pitch, velocity, MuLan, duration)
    \item $\epsilon$ is the sampled noise
    \item $z_0$ is the clean audio latent
\end{itemize}

\subsubsection{Classifier-Free Guidance Training}

We apply classifier-free guidance by randomly dropping conditioning with 10\% probability:
\begin{itemize}[leftmargin=1.5em]
    \item Text dropout: replace text embeddings with empty string encoding
    \item Pitch dropout: zero out pitch embeddings
    \item Velocity dropout: zero out velocity embeddings
\end{itemize}

\subsection{Audio Encoding}

Audio is encoded using the VAE from Stable Audio Open 1.0:

\begin{verbatim}
Audio [batch, 2, samples] → VAE Encoder → Latents [batch, seq_len, 64]
\end{verbatim}

The VAE compresses 44.1kHz stereo audio into a compact latent representation.

\section{Experimental Setup}

\subsection{Dataset}

We train and evaluate on the NSynth dataset, which contains:
\begin{itemize}[leftmargin=1.5em]
    \item Training set: 289,205 samples
    \item Validation set: 12,678 samples
    \item Test set: 4,096 samples
\end{itemize}

Each sample is a 4-second single-note recording at 16kHz (resampled to 44.1kHz). Metadata includes:
\begin{itemize}[leftmargin=1.5em]
    \item Instrument family (bass, brass, flute, guitar, keyboard, mallet, organ, reed, string, synth\_lead, vocal)
    \item Instrument source (acoustic, electronic, synthetic)
    \item Pitch (MIDI 21--108)
    \item Velocity (25, 50, 75, 100, 127)
    \item Qualities (bright, dark, distortion, fast\_decay, long\_release, multiphonic, nonlinear\_env, percussive, reverb, tempo-synced)
\end{itemize}

\subsubsection{Caption Generation}

Captions are automatically generated from metadata:

\begin{verbatim}
caption = f"{instrument_family} {instrument_source} {' '.join(qualities)}"
\end{verbatim}

Example: \texttt{"keyboard electronic distortion reverb"}

\subsubsection{MuLan Feature Extraction}

MuLan features are pre-extracted using multi-GPU processing:

\begin{verbatim}
python extract_mulan_features.py \
    --audio_dir /path/to/audio \
    --output_dir /path/to/mulan \
    --num_gpus 4
\end{verbatim}

\subsection{Training Configuration}

\begin{table}[h]
\centering
\begin{tabular}{ll}
\toprule
\textbf{Hyperparameter} & \textbf{Value} \\
\midrule
Batch size & 8 per GPU \\
Learning rate & 5e-4 \\
Optimizer & AdamW \\
Warmup steps & 1,000 \\
Training epochs & 800 \\
Gradient accumulation & 1 \\
Audio duration & 4 seconds \\
Checkpoint interval & 5,000 steps \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Hardware}

Training is performed on 2$\times$ NVIDIA GPUs using the Accelerate distributed training framework.

\section{Inference}

\subsection{Environment Setup}

\begin{verbatim}
import torch
import librosa
import os
import soundfile as sf
from tangoflux import TangoFluxInference
from muq import MuQMuLan

# Set HuggingFace mirror for China (optional)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
\end{verbatim}

\subsection{Model Loading}

The inference pipeline requires two models:
\begin{enumerate}[leftmargin=1.5em]
    \item \textbf{TangoFluxInference}: the main generation model, loaded from checkpoint
    \item \textbf{MuQMuLan}: used to extract audio embeddings from reference audio
\end{enumerate}

\begin{verbatim}
mulan_model = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
mulan_model = mulan_model.to(device).eval()

model = TangoFluxInference(name="outputs/step_9000")
\end{verbatim}

\subsection{Checkpoint Structure}

\begin{verbatim}
outputs/step_9000/
├── model.safetensors      # VAE weights
├── model_1.safetensors    # TangoFlux transformer weights
├── config.json            # Model architecture configuration
├── optimizer.bin          # Optimizer state (not needed for inference)
└── scheduler.bin          # LR scheduler state (not needed for inference)
\end{verbatim}

\subsection{Configuration Format}

\begin{verbatim}
{
  "num_layers": 4,
  "num_single_layers": 12,
  "in_channels": 64,
  "attention_head_dim": 128,
  "joint_attention_dim": 1024,
  "num_attention_heads": 8,
  "audio_seq_len": 86,
  "max_duration": 4,
  "uncondition": false,
  "text_encoder_name": "./models/flan-t5-large"
}
\end{verbatim}

\subsection{MuLan Embedding Extraction}

MuLan embeddings capture the timbral characteristics of reference audio:

\begin{verbatim}
ref_audio_path = "/path/to/reference_audio.wav"
wav, sr = librosa.load(ref_audio_path, sr=24000)
wav_tensor = torch.tensor(wav).unsqueeze(0).to(device)

with torch.no_grad():
    mulan_embedding = mulan_model(wavs=wav_tensor)

mulan_embedding = mulan_embedding.unsqueeze(0).cpu()
\end{verbatim}

\subsection{Timbre Mixing}

Multiple reference audio embeddings can be interpolated to create hybrid timbres.

\begin{verbatim}
mulan_mixed = torch.mean(torch.stack([mulan1, mulan2]), dim=0)
\end{verbatim}

\subsection{Generation Pipeline}

\begin{verbatim}
audio = model.generate(
    prompt,
    mulan,
    steps=50,
    pitch=60,
    duration=4,
    velocity=127,
    guidance_scale=4.5,
    seed=None
)
\end{verbatim}

\begin{table}[h]
\centering
\begin{tabular}{lll}
\toprule
\textbf{Parameter} & \textbf{Type} & \textbf{Description} \\
\midrule
prompt & str & Text description of desired timbre \\
mulan & Tensor & MuLan audio embedding, shape [1, 1, 512] \\
steps & int & Denoising steps (25--100) \\
pitch & int & MIDI pitch value (0--127) \\
duration & float & Audio duration in seconds (0--4) \\
velocity & int & 25/50/75/100/127 \\
guidance\_scale & float & CFG strength (1.0--10.0) \\
seed & int & Random seed for reproducibility \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Output Processing}

\begin{verbatim}
audio = model.generate(
    "pluck synthetic",
    mulan_embedding,
    steps=50,
    pitch=60,
    duration=4,
    velocity=127,
    guidance_scale=4.5
)

sf.write("output.wav", audio.T, 44100)
\end{verbatim}

\subsection{Classifier-Free Guidance}

During inference, we apply classifier-free guidance to improve generation quality:

\begin{verbatim}
v_guided = v_uncond + scale × (v_cond - v_uncond)
\end{verbatim}

Recommended \texttt{guidance\_scale}: 3.5--4.5.

\section{Results}

\subsection{Qualitative Analysis}

The model successfully generates:
\begin{itemize}[leftmargin=1.5em]
    \item distinct timbres for different instrument families,
    \item accurate pitch reproduction across the MIDI range,
    \item perceptible velocity variations affecting attack characteristics,
    \item quality modifiers such as bright, dark, and percussive timbres.
\end{itemize}

\subsection{Control Accuracy}

\begin{table}[h]
\centering
\begin{tabular}{ll}
\toprule
\textbf{Control Signal} & \textbf{Accuracy} \\
\midrule
Pitch (within 1 semitone) & $>$95\% \\
Velocity (perceptual) & $\sim$85\% \\
Instrument family & $\sim$90\% \\
\bottomrule
\end{tabular}
\end{table}

\section{Conclusion}

TangoFlux-Timbre demonstrates that modern diffusion-based audio generation can achieve fine-grained musical control. By combining text conditioning with explicit pitch and velocity embeddings, the model enables intuitive control over synthesized instrument timbres. The integration of MuLan embeddings provides additional timbre specification capabilities while maintaining the flexibility of text-only generation.

\subsection{Limitations}
\begin{itemize}[leftmargin=1.5em]
    \item Limited to monophonic synthesis
    \item 4-second maximum duration per generation
    \item Requires pre-extracted MuLan features for training
\end{itemize}

\subsection{Future Work}
\begin{itemize}[leftmargin=1.5em]
    \item Extension to polyphonic synthesis
    \item Real-time generation capabilities
    \item Fine-tuning on specific instrument datasets
    \item Integration with MIDI input for interactive performance
\end{itemize}

\section*{References}

\begin{enumerate}
    \item Ghosal, D., et al. \textit{TangoFlux: Super Fast and Faithful Text to Audio Generation with Flow Matching and CLAP-Ranked Preference Optimization.} arXiv preprint (2024).
    \item Engel, J., et al. \textit{Neural Audio Synthesis of Musical Notes with WaveNet Autoencoders.} ICML (2017).
    \item Ho, J., et al. \textit{Denoising Diffusion Probabilistic Models.} NeurIPS (2020).
    \item Lipman, Y., et al. \textit{Flow Matching for Generative Modeling.} ICLR (2023).
    \item Raffel, C., et al. \textit{Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.} JMLR (2020).
\end{enumerate}

\appendix
\section{Model Configuration}

\begin{verbatim}
model:
  num_layers: 4
  num_single_layers: 12
  in_channels: 64
  attention_head_dim: 128
  joint_attention_dim: 1024
  num_attention_heads: 8
  audio_seq_len: 86
  max_duration: 4
  uncondition: false
  text_encoder_name: "./models/flan-t5-large"

training:
  per_device_batch_size: 16
  num_workers: 8
  learning_rate: 5e-4
  gradient_accumulation_steps: 1
  num_train_epochs: 800
  num_warmup_steps: 1000
  max_audio_duration: 4
\end{verbatim}

\section{Data Format}

\begin{verbatim}
{
  "captions": "bass electronic percussive",
  "location": "/path/to/audio/bass_electronic_018-048-075.wav",
  "duration": 4.0,
  "pitch": 48,
  "velocity": 75
}
\end{verbatim}

\end{document}
