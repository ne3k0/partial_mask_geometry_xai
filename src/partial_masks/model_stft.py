# model_stft.py — model-specific STFT and mel computation

import warnings

import numpy as np
import torch
import librosa
from scipy.signal import istft as _scipy_istft, lfilter, butter, sosfiltfilt


class PannsSTFTHelper:
    """
    PANNs CNN14 uses torchlibrosa STFT + librosa mel filterbank + 10*log10 compression.
    """

    def __init__(self, model):
        # support both panns_inference wrapper (model.at.model) and direct model (model.model)
        if hasattr(model, 'at'):
            inner_model = model.at.model
        else:
            inner_model = model.model
        if hasattr(inner_model, 'module'):  # DataParallel wraps the model
            inner_model = inner_model.module
        self.stft_fn = inner_model.spectrogram_extractor.stft
        self.device = next(inner_model.parameters()).device
        self.sr = 32000
        self.n_fft = 1024
        self.frame_length = 1024  # same as n_fft for PANNs
        self.hop = 320
        self.n_mels = 64
        self.x = 100
        self.fmin = 50
        self.fmax = 14000
        self.W = librosa.filters.mel(sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax)
        self.M2F = [np.where(self.W[k] > 0)[0] for k in range(self.n_mels)]
        self.melW = torch.tensor(self.W.T, dtype=torch.float32)  # (513, 64)

    def compute_stft(self, signal):
        """
        Compute STFT using torchlibrosa. Returns complex numpy array shape (n_fft_bins, T).
        """

        audio_t = torch.tensor(signal[None, :], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            real, imag = self.stft_fn(audio_t)
        # real, imag: (1, 1, T, 513) → squeeze and transpose to (513, T)
        real = real.squeeze(0).squeeze(0).cpu().numpy().T  # (513, T)
        imag = imag.squeeze(0).squeeze(0).cpu().numpy().T  # (513, T)
        return real + 1j * imag

    def to_logmel(self, Zxx_perturbed):
        """
        Perturbed STFT -> power -> mel -> 10*log10. Returns log mel shape (n_mels, T).
        """
        power = np.abs(Zxx_perturbed) ** 2
        mel = self.W @ power
        return 10.0 * np.log10(np.maximum(mel, 1e-10))

    def istft(self, Zxx, length=None):
        return librosa.istft(Zxx, hop_length=self.hop, win_length=self.frame_length, n_fft=self.n_fft, window='hann', center=True, length=length)


class AstSTFTHelper:
    """
    AST uses torchaudio kaldi STFT + kaldi mel banks + natural log compression.
    """

    def __init__(self):
        from transformers import AutoFeatureExtractor
        from torchaudio.compliance.kaldi import _get_waveform_and_window_properties, _get_window, get_mel_banks

        self._get_waveform_and_window_properties = _get_waveform_and_window_properties
        self._get_window = _get_window

        self.sr = 16000
        self.n_fft = 512  # kaldi pads 400 → 512
        self.frame_length = 400
        self.hop = 160
        self.preemphasis_coefficient = 0.97  # kaldi default, needed for depreemphasis in audio reconstruction
        self.n_mels = 128
        self.x = 101
        self.n_fft_bins = self.n_fft // 2 + 1  # 257

        fe = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.max_length = fe.max_length  # 1024
        self.fe_mean = fe.mean
        self.fe_std = fe.std

        # kaldi mel banks: shape (128, 256), then pad right -> (128, 257)
        mel_banks, _ = get_mel_banks(self.n_mels, self.n_fft, self.sr, low_freq=20.0, high_freq=self.sr / 2.0, vtln_low=100.0, vtln_high=-500.0, vtln_warp_factor=1.0)
        self.mel_banks = torch.nn.functional.pad(mel_banks, (0, 1), mode='constant', value=0) # (128, 257) — includes DC bin position (zeros)

        # W for perturbation.py: (n_mels, n_fft_bins) = (128, 257)
        self.W = self.mel_banks.numpy()
        self.M2F = [np.where(self.W[k] > 0)[0] for k in range(self.n_mels)]

    def compute_stft(self, signal):
        """
        Compute STFT using kaldi internals (AST's own implementation). Returns complex numpy array shape (n_fft_bins, T).
        """

        waveform = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)

        result = self._get_waveform_and_window_properties(waveform, channel=0, sample_frequency=self.sr, frame_shift=10.0, frame_length=25.0, round_to_power_of_two=True, preemphasis_coefficient=0.97)
        processed_waveform = result[0]
        padded_window_size = result[-1]

        strided_input, _ = self._get_window(processed_waveform, padded_window_size, self.frame_length, self.hop, window_type='hanning', blackman_coeff=0.42, snip_edges=True, raw_energy=True, energy_floor=0.0, dither=0.0, remove_dc_offset=True, preemphasis_coefficient=0.97)

        kaldi_stft = torch.fft.rfft(strided_input, n=padded_window_size) # shape (T, 257) complex → transpose to (257, T)
        return kaldi_stft.numpy().T

    def to_logmel(self, Zxx_perturbed):
        """
        Perturbed STFT -> power -> kaldi mel -> log. Returns log mel shape (n_mels, T).
        """
        power = np.abs(Zxx_perturbed) ** 2
        power_t = torch.tensor(power.T, dtype=torch.float32)
        mel_banks_f = self.mel_banks.to(torch.float32)
        mel = torch.mm(power_t, mel_banks_f.T)
        eps = torch.tensor(torch.finfo(torch.float32).eps)
        log_mel = torch.max(mel, eps).log()
        return log_mel.numpy().T

    def istft(self, Zxx, length=None):
        win = np.hanning(self.frame_length)
        with warnings.catch_warnings():
            # snip_edges=True means the forward added no boundary padding, so scipy warns that edge samples violate NOLA. Interior is fine.
            warnings.simplefilter("ignore", UserWarning)
            _, audio = _scipy_istft(Zxx, fs=self.sr, window=win,
                                    nperseg=self.frame_length,
                                    noverlap=self.frame_length - self.hop,
                                    nfft=self.n_fft, boundary=False)
        audio = lfilter([1.0], [1.0, -self.preemphasis_coefficient], audio)
        sos = butter(2, 20.0, btype="highpass", fs=self.sr, output="sos")
        audio = sosfiltfilt(sos, audio)
        return audio[:length] if length is not None else audio


def get_stft_helper(model_id, model=None):
    """
    Function returning the right STFT helper for the model.
    """

    if model_id.startswith("PANNs/"):
        return PannsSTFTHelper(model)
    elif model_id == "MIT/ast-finetuned-audioset-10-10-0.4593":
        return AstSTFTHelper()
    else:
        raise ValueError(f"Unsupported model_id: {model_id}")
