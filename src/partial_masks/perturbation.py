# perturbation.py — mask generation, binary boundary expansion, and fill strategies for complex-STFT perturbation of mel-spectrogram audio classifiers.

# Binary boundaries only: smoothed boundaries (Gaussian/Gabor) were shown to have no significant effect on attribution and so were removed. 

import numpy as np
from numba import njit, prange

from .config import K_F, K_T, L, P, GAUSSIAN_NOISE_SEED


# Fill strategies


def zero_fill(Zxx, stft_mask):
    return Zxx * stft_mask  # complex stft either on (1) or off (0)


def mean_fill(Zxx, stft_mask):
    mean_mag = np.mean(np.abs(Zxx), axis=1, keepdims=True)  # (F, 1) time-averaged magnitude per FFT bin
    fill = mean_mag * np.exp(1j * np.angle(Zxx)) # original phase, mean magnitude
    return stft_mask * Zxx + (1 - stft_mask) * fill


def make_gaussian_noise_fill(Zxx):
    rng = np.random.default_rng(GAUSSIAN_NOISE_SEED)
    mean_mag = np.mean(np.abs(Zxx), axis=1, keepdims=True)
    noise_mag = np.abs(rng.normal(0, mean_mag, Zxx.shape))
    Zxx_noise = noise_mag * np.exp(1j * np.angle(Zxx))  # preserve original phase
    def gaussian_noise_fill(Zxx, stft_mask):
        return stft_mask * Zxx + (1 - stft_mask) * Zxx_noise
    return gaussian_noise_fill


FILL_FNS = {
    "zero": zero_fill,
    "mean": mean_fill,
    "gaussian_noise": lambda Zxx, mask: make_gaussian_noise_fill(Zxx)(Zxx, mask),
}


# Binary boundary: mel-cell masks → STFT-space masks


def pack_ragged_arrays(list_of_arrays):
    # convert list of variable-length arrays to 2D array (padded) + lengths
    max_len = max(len(a) for a in list_of_arrays)
    arr = np.full((len(list_of_arrays), max_len), -1, dtype=np.int64)
    lens = np.zeros(len(list_of_arrays), dtype=np.int64)
    for i, a in enumerate(list_of_arrays):
        arr[i, :len(a)] = a
        lens[i] = len(a)
    return arr, lens


@njit(parallel=True)
def expand_binary(masks_flat, M, X, M2F_arr, M2F_len, X2T_arr, X2T_len, F, T, L):
    # Hard rectangular expansion: ON cells → all corresponding FFT bins × frames = 1.0
    stft_masks = np.zeros((L, F, T))
    for i in prange(L):
        for m in range(M):
            for t in range(X):
                if masks_flat[i, m * X + t]:
                    for fi in range(M2F_len[m]):
                        fft_bin = M2F_arr[m, fi]
                        for ti in range(X2T_len[t]):
                            frame = X2T_arr[t, ti]
                            stft_masks[i, fft_bin, frame] = 1.0
    return stft_masks


def boundary_binary(masks, W, M2F, M, X, X2T, F, T, **kwargs):
    M2F_arr, M2F_len = pack_ragged_arrays(M2F)
    X2T_arr, X2T_len = pack_ragged_arrays(X2T)
    L = masks.shape[0]
    return expand_binary(masks, M, X, M2F_arr, M2F_len, X2T_arr, X2T_len, F, T, L)


# Segmentation: random block boundaries per seed, shared across samples


def spaced_splits(rng, n, K, min_band):
    """
    Generate K-1 split points in [0, n) with minimum band size min_band.
    Returns sorted array of split positions. All K bands are >= min_band.
    """
    n_slots = n - K * min_band
    if n_slots < K - 1:
        raise ValueError("Cannot fit %d bands of min %d in %d" % (K, min_band, n))
    gaps = np.sort(rng.choice(np.arange(1, n_slots + 1), size=K - 1, replace=False))
    return gaps + np.arange(1, K) * min_band


def generate_block_boundaries(rng, M, X, K_f=K_F, K_t=K_T, min_freq_band=3, min_time_band=3):
    """
    Returns block_map: int32 array shape (M, X), values 0..K-1.
    """

    freq_splits = spaced_splits(rng, M, K_f, min_freq_band)
    freq_bands = np.searchsorted(freq_splits, np.arange(M))

    time_offset = rng.integers(0, X)
    time_splits = spaced_splits(rng, X, K_t, min_time_band)
    rolled_bands = np.searchsorted(time_splits, np.arange(X))
    dest = (np.arange(X) + time_offset) % X
    time_bands = np.empty(X, dtype=np.int32)
    time_bands[dest] = rolled_bands

    block_map = (freq_bands[:, np.newaxis] * K_t + time_bands[np.newaxis, :]).astype(np.int32)
    return block_map  # shape (M, X)


# Bernoulli sampling: build masks for each sample based on block_map


@njit
def build_cell_masks(block_map_flat, draws):
    """
    Returns masks: 2d bool array shape (L, n_cells).
    """
    L = draws.shape[0]
    masks = np.zeros((L, len(block_map_flat)), dtype=np.bool_)
    for i in range(L):
        for j in range(len(block_map_flat)):
            masks[i, j] = draws[i, block_map_flat[j]]
    return masks
