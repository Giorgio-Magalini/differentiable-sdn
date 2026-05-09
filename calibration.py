import numpy as np
import torch
from scipy.signal import find_peaks
from utils import load_positions, load_homula_rirs
from pathlib import Path

def _parabolic_interpolation(signal, peak_index):
    """
    Perform 3-point parabolic interpolation around a discrete peak.
    Returns a fractional peak index.
    """
    if peak_index <= 0 or peak_index >= len(signal) - 1:
        return float(peak_index)

    y_m1 = signal[peak_index - 1]
    y_0  = signal[peak_index]
    y_p1 = signal[peak_index + 1]

    denominator = (y_m1 - 2 * y_0 + y_p1)

    if denominator == 0:
        return float(peak_index)

    delta = 0.5 * (y_m1 - y_p1) / denominator
    return float(peak_index + delta)

def _select_topk_earliest_peak(signal, K):
    """
    Select the earliest peak among the top-K highest amplitude peaks.
    """
    peaks, _ = find_peaks(signal)

    if peaks.size == 0:
        # fallback to return index of max sample
        return int(np.argmax(np.abs(signal)))

    peak_amplitudes = signal[peaks]
    sorted_indices = np.argsort(peak_amplitudes)[::-1]
    topk_peaks = peaks[sorted_indices[:K]]

    return int(np.min(topk_peaks))

def _get_top_k_peaks(signal: np.ndarray, K: int, search_window_samples: int = 5) -> int:
    """
    Finds positive and negative peaks within a temporal window around the
    global absolute maximum, then returns the top K indices ordered by magnitude.
    """
    # 1. Find the global absolute maximum as a reference point
    abs_signal = np.abs(signal)
    global_max_idx = np.argmax(abs_signal)

    # 2. Define search boundaries (Region of Interest)
    start_idx = max(0, global_max_idx - search_window_samples)
    end_idx = min(len(signal), global_max_idx + 2)

    roi = signal[start_idx:end_idx]

    # 3. Detect peaks within the ROI
    # We use a small height threshold relative to the ROI max to filter noise
    roi_threshold = 0.33 * np.max(np.abs(roi))

    pos_peaks, _ = find_peaks(roi, height=roi_threshold)
    neg_peaks, _ = find_peaks(-roi, height=roi_threshold)

    # Combine and shift indices back to the original signal coordinates
    detected_peaks = np.concatenate([pos_peaks, neg_peaks]) + start_idx

    # Ensure the global maximum is included even if find_peaks missed it
    if global_max_idx not in detected_peaks:
        detected_peaks = np.append(detected_peaks, global_max_idx)

    # 4. Sort all detected peaks by absolute magnitude (descending)
    peak_amplitudes = np.abs(signal[detected_peaks])
    top_indices = np.argsort(peak_amplitudes)[-K:][::-1]

    return int(np.min(detected_peaks[top_indices]))


from scipy.signal import envelope

def _extract_toas(
    rirs,
    sr=48000,
    K=4,
    enable_true_peak=True
):
    """
    Estimate Time of Arrivals (TOAs) from a multichannel RIR.

    Parameters
    ----------
    rirs : np.ndarray

    sr : int
        Sampling rate.
    K : int
        Number of peaks considered in 'topk_earliest_peak'.
    enable_true_peak : bool
        If True, apply parabolic interpolation refinement.

    Returns
    -------
    np.ndarray
        Array of TOAs (seconds) for each channel.
    """
    num_sources = rirs.shape[2]
    num_channels = rirs.shape[1]
    toas = np.zeros((num_channels, num_sources))

    for src_idx in range(num_sources):
        for channel_idx in range(num_channels):

            signal = rirs[:, channel_idx, src_idx]

            signal = envelope(signal)

            peak_index = int(np.argmax(np.abs(signal)))

            if enable_true_peak:
                peak_index = _parabolic_interpolation(signal, int(peak_index))

            toas[channel_idx, src_idx] = peak_index / sr

    return toas  # (M, K)


def estimate_multiple_sources_shared_delay(
    mic_positions,
    toas_matrix,
    original_source_positions,
    c=343.0
):
    """
    Estimate positions of K sources simultaneously with a single shared delay.

    Parameters
    ----------
    mic_positions : (M, 2) array
        Microphone coordinates.
    toas_matrix : (M, K) array
        TOAs for each microphone (rows) and each source (columns).
    original_source_positions : (K, 2) array
        Ground truth source coordinates used only to determine side.
    c : float
        Speed of sound.

    Returns
    -------
    positions : (K, 2) array
        Estimated source positions.
    delta_hat : float
        Shared global delay (meters).
    """

    mic_positions = np.asarray(mic_positions, dtype=np.float64)
    toas_matrix = np.asarray(toas_matrix, dtype=np.float64)
    original_source_positions = np.asarray(original_source_positions, dtype=np.float64)

    M, K = toas_matrix.shape
    distances = toas_matrix * c

    # ---- Rotation to align ULA ----
    p0 = mic_positions[0]
    pN = mic_positions[-1]

    direction = pN - p0
    norm_dir = np.linalg.norm(direction)
    if norm_dir == 0:
        raise ValueError("Degenerate array configuration")

    e1 = direction / norm_dir
    e2 = np.array([-e1[1], e1[0]])
    rotation_matrix = np.vstack([e1, e2])

    mic_rotated = (rotation_matrix @ (mic_positions - p0).T).T
    y_array = np.mean(mic_rotated[:, 1])

    # Rotate original source positions to determine side
    src_rotated = (rotation_matrix @ (original_source_positions - p0).T).T

    # ---- Build global linear system ----
    num_unknowns = 2 * K + 1
    A = np.zeros((M * K, num_unknowns))
    b_vector = np.zeros(M * K)

    row_index = 0

    for k in range(K):
        for i in range(M):
            mic_x = mic_rotated[i, 0]
            distance_ik = distances[i, k]

            col_xk = 2 * k
            col_ck = 2 * k + 1
            col_delta = 2 * K

            A[row_index, col_xk] = -2.0 * mic_x
            A[row_index, col_ck] = 1.0
            A[row_index, col_delta] = 2.0 * distance_ik

            b_vector[row_index] = distance_ik**2 - mic_x**2

            row_index += 1

    AtA = A.T @ A
    Atb = A.T @ b_vector

    if abs(np.linalg.det(AtA)) < 1e-12:
        raise ValueError("Ill-conditioned global system")

    solution = np.linalg.inv(AtA) @ Atb
    delta_hat = solution[2 * K]

    estimated_positions = np.zeros((K, 2))

    # ---- Recover each source position ----
    for k in range(K):
        xk = solution[2 * k]
        ck = solution[2 * k + 1]

        y_relative_squared = ck + delta_hat**2 - xk**2
        y_relative_squared = max(y_relative_squared, 0.0)
        y_relative = np.sqrt(y_relative_squared)

        # Determine side from original rotated position
        if src_rotated[k, 1] >= y_array:
            yk_rot = y_array + abs(y_relative)
        else:
            yk_rot = y_array - abs(y_relative)

        estimated_rotated = np.array([xk, yk_rot])
        estimated_positions[k] = rotation_matrix.T @ estimated_rotated + p0

    return estimated_positions, delta_hat

def rescale_rirs_to_unit_excitation(
    rirs: torch.Tensor,
    mic_positions: torch.Tensor,
    src_positions: torch.Tensor,
) -> torch.Tensor:
    """
    Rescale RIRs to be consistent with a unit-impulse excitation of the SDN.

    For each source k, the reference microphone is selected as the one with
    the highest peak amplitude (best SNR). The scaling factor is:

        gain_k = g_SM[ref_k] / max(|RIR[ref_k]|)

    where g_SM[ref_k] = 1 / d(src_k, mic_ref_k) is the direct-path gain at
    the reference microphone (SDN spherical spreading convention).

    A single scalar gain_k is applied uniformly to all channels of source k,
    preserving inter-channel amplitude ratios and dataset normalization
    homogeneity.

    Parameters
    ----------
    rirs : torch.Tensor, shape (K, M, T)
        Multichannel RIRs, one per source.
    mic_positions : torch.Tensor, shape (M, 3)
        Microphone 3-D coordinates [m].
    src_positions : torch.Tensor, shape (K, 3)
        Source 3-D coordinates [m].

    Returns
    -------
    torch.Tensor, shape (K, M, T)
        Rescaled RIRs.
    """
    rirs_out = rirs.clone()

    for k in range(src_positions.shape[0]):
        # Peak amplitude per microphone for source k: (M,)
        peak_per_mic = torch.amax(torch.abs(rirs[k]), dim=-1)

        # Reference mic: highest peak amplitude
        ref_mic = int(torch.argmax(peak_per_mic).item())

        # Direct-path gain at reference mic (SDN convention: 1/d)
        d_ref = torch.norm(src_positions[k] - mic_positions[ref_mic]).item()
        if d_ref < 1e-6:
            raise ValueError(
                f"Source {k} is coincident with reference microphone (mic {ref_mic})."
            )
        g_SM_ref = 1.0 / d_ref

        # Combined scale factor
        peak_ref        = peak_per_mic[ref_mic].item()
        gain            = g_SM_ref / peak_ref
        rirs_out[k]     = rirs[k] * gain

    return rirs_out

def _apply_delay(rirs, delay_samples):
    """
    Cut or zero-pad the beginning of RIRs by delay_samples.
    Positive delay  →  cut leading samples (source was farther than expected).
    Negative delay  →  prepend zeros (source was closer than expected).
    """
    S = rirs.shape[0]
    M = rirs.shape[1]

    if delay_samples > 0:
        return rirs[:, :, delay_samples:]
    elif delay_samples < 0:
        pad = torch.zeros(S, M, -delay_samples, dtype=rirs.dtype)
        return torch.cat([pad, rirs], dim=1)
    else:
        return rirs

def load_and_calibration_pipeline(
    rirs_path,
    mic_pos_path,
    src_pos_path,
    c=343.0,
    sr=48000,
    flip: bool = False,
):
    rirs_paths = [Path(p) for p in rirs_path]
    src_pos_path = Path(src_pos_path)

    calibrated_src_pos_path = src_pos_path.with_stem(
        src_pos_path.stem + "_calibrated"
    )

    # One delay file shared across the system
    delay_path = rirs_paths[0].with_stem(
        rirs_paths[0].stem + "_system_delay_meters"
    ).with_suffix(".txt")

    # # ------------------------------------------------------------------ #
    # # Step 1: If delay and calibrated positions already exist, load and apply
    # # ------------------------------------------------------------------ #
    # if delay_path.exists() and calibrated_src_pos_path.exists():
    #     mic_positions            = load_positions(mic_pos_path)
    #     delay_meters             = float(open(delay_path).read().strip())
    #     delay_samples            = int(round(delay_meters / c * sr))
    #     rirs                     = load_homula_rirs(rirs_path, sr=sr, trim=True, flip=flip)
    #     calibrated_src_positions = load_positions(calibrated_src_pos_path)
    #     calibrated_rirs          = _apply_delay(rirs, delay_samples)
    #     calibrated_rirs = rescale_rirs_to_unit_excitation(
    #         calibrated_rirs, mic_positions, calibrated_src_positions
    #     )
    #     return calibrated_rirs, calibrated_src_positions, mic_positions

    # ------------------------------------------------------------------ #
    # Step 2: Load RIRs, mic positions, and source positions
    # ------------------------------------------------------------------ #
    rirs          = load_homula_rirs(rirs_path, sr=sr, trim=True, flip=flip)   # (M, T)
    mic_positions = load_positions(mic_pos_path)          # (M, 3)
    src_positions = load_positions(src_pos_path)          # (K, 3)

    mic_pos_np = mic_positions.numpy()
    src_pos_np = src_positions.numpy()

    # ------------------------------------------------------------------ #
    # Step 3: Extract TOAs
    # ------------------------------------------------------------------ #
    rirs_np = rirs.numpy().T                  # (T, M)
    toas    = _extract_toas(rirs_np, sr=sr)    # (M,)

    # ------------------------------------------------------------------ #
    # Step 4: Estimate source positions and shared system delay
    # ------------------------------------------------------------------ #
    K          = src_pos_np.shape[0]
    mic_pos_2d = mic_pos_np[:, :2]
    src_pos_2d = src_pos_np[:, :2]
    toas_matrix = toas.reshape(-1, K)         # (M, K)

    estimated_positions_2d, delta_hat = estimate_multiple_sources_shared_delay(
        mic_pos_2d, toas_matrix, src_pos_2d, c=c
    )

    # Reconstruct 3D positions reusing original z coordinate
    estimated_positions_3d = np.hstack([
        estimated_positions_2d,
        src_pos_np[:, 2:3]
    ])

    # Save calibrated source positions
    with open(calibrated_src_pos_path, 'w') as f:
        f.write("x,y,z\n")
        for pos in estimated_positions_3d:
            f.write(f"{pos[0]:.6f},{pos[1]:.6f},{pos[2]:.6f}\n")

    # ------------------------------------------------------------------ #
    # Step 5: Compute delay in samples and save to file
    # ------------------------------------------------------------------ #
    with open(delay_path, 'w') as f:
        f.write(f"{delta_hat:.12f}")

    delay_samples = int(round(delta_hat / c * sr))

    # ------------------------------------------------------------------ #
    # Step 6: Apply delay to original RIRs and return
    # ------------------------------------------------------------------ #
    calibrated_rirs          = _apply_delay(rirs, delay_samples)
    calibrated_src_positions = torch.tensor(estimated_positions_3d, dtype=torch.float32)
    calibrated_rirs = rescale_rirs_to_unit_excitation(
        calibrated_rirs, mic_positions, calibrated_src_positions
    )

    return calibrated_rirs, calibrated_src_positions, mic_positions