# Differentiable Scattering Delay Networks

**Room Characterization and Novel-View Acoustics Synthesis With Differentiable Scattering Delay Networks**

> Master's thesis project — Politecnico di Milano  
> Supervisor: Prof. Alessandro Ilic Mezza

---

## Overview

This repository implements a **differentiable Scattering Delay Network (SDN)** for data-driven room acoustics modelling. The model learns to synthesize multichannel Room Impulse Responses (RIRs) from measured data by backpropagating through a physically-grounded reverberator via full BPTT.

Key contributions:
- **SE-SDN**: Scattering-Equation SDN with trainable Householder scattering junctions and FIR wall filters
- **Physical calibration pipeline**: Automatic TOA-based source position refinement and system delay estimation
- **Novel-view synthesis**: Generalization to unseen microphone positions at inference time
- **Acoustic loss suite**: EDC, mel-EDR, and SoftEDP losses for perceptually meaningful training

---

## Repository Structure

```
.
├── config/
│   ├── homula-rir/         # Training configs for real RIR dataset
│   └── rooms/              # Room geometry configs
├── data/
│   └── homula-rir/         # ULA positions, source positions, RIR WAVs
│   └── simulated/          # Simulated RIRs 
├── saved_models/           # Checkpoints (gitignored)
├── figures/                # Output figures (gitignored)
├── sdn.py                  # SDN model
├── junction.py             # Householder scattering junction
├── integer_delay.py        # Batch-aware integer delay lines
├── curves.py               # EDC / mel-EDR / EDP curve computations
├── losses.py               # Loss functions
├── calibration.py          # TOA extraction and calibration pipeline
├── position_utils.py       # Geometric utilities (node positions, distances)
├── train_sdn.py            # Training entry point
└── utils.py                # I/O utilities
```

---

## Architecture

The **SE-SDN** models a rectangular room with `N = 6` wall junctions. Each junction implements a trainable **Householder scattering matrix** parameterized by a learnable admittance vector. Delay lines route signals between the source, junctions, and microphone via integer-sample delays derived from room geometry.

**Trainable parameters:**
| Component | Parameter | Description |
|---|---|---|
| `HouseholderScatteringJunction` | `weight` (N−1,) | Per-junction admittance vector |
| `SDN` | `junction_filters` (N, fir_order+1) | Per-wall FIR absorption filters |
| `SDN` | `pressure_weights` (N·(N−1),) | Microphone pressure extraction weights |

**Signal flow:**

```
x[k]  →  src_to_nodes  →  pp  →  Householder junctions  →  pm
                                                              ↓
y[k]  ←  nodes_to_mic  ←  junction_filters  ←  pressure_weights * pm
      +  direct_gain   ←  src_to_mic
```

---

## Calibration Pipeline

The pipeline in `calibration.py` performs the following steps automatically:

1. **TOA extraction** — envelope-based peak detection with parabolic sub-sample interpolation
2. **Source position estimation** — joint least-squares solver for `K` sources and a shared system delay `δ`
3. **RIR alignment** — trim leading samples by `δ / c · sr`
4. **Amplitude rescaling** — normalize each source to unit-impulse excitation (SDN convention: `1/d` direct-path gain)

Calibration artifacts are written alongside the input data:
- `*_calibrated.csv` — refined source positions (x, y, z)
- `*_system_delay_meters.txt` — shared system delay in meters (sample-rate-independent)

---

## Loss Functions

All losses operate natively on batched RIRs `(B, T)`.

| Loss | Class | Description |
|---|---|---|
| EDC | `EDCLoss` | MSE on full-band Schroeder energy decay curves |
| mel-EDR | `MelEDRLogLoss` | MAE on log-amplitude mel-frequency energy decay reliefs |
| SoftEDP | `EDPLoss` | MSE on sigmoid-approximated echo density profiles |

The weighted total loss is:

```
L = λ_edc · L_EDC + λ_edr · L_EDR + λ_edp · L_EDP
```

Loss weights are set in the training config under `training.lambda_*`.

---

## Installation

```bash
conda create -n differentiable_sdn python=3.10
conda activate differentiable_sdn
pip install torch torchaudio numpy scipy matplotlib pyroomacoustics pyyaml tqdm
```

---

## Training

```bash
python train_sdn.py \
    --config config/homula-rir/sdn_fir6_16kHz.yaml \
    --room   config/rooms/schiavoni_room.yaml \
    --device cuda:0
```

**Config fields** (see `config/` for full examples):

| Field | Description |
|---|---|
| `sr` | Sampling rate (recommended: 16000 Hz) |
| `training.split_mode` | Train/val microphone split strategy (see below) |
| `training.batch_size` | Microphones per forward pass |
| `training.accumulation_factor` | Gradient accumulation steps |
| `sdn.fir_order` | FIR wall filter order |
| `sdn.alpha` | Initial wall absorption coefficient |

### Split Modes

| Mode | Description |
|---|---|
| `even` | Even-indexed mics → train, odd-indexed → val |
| `first_half` | First N/2 mics → train, second N/2 → val |
| `second_half` | Second N/2 mics → train, first N/2 → val |
| `center_line_x` | Horizontal center row → train *(simulated only)* |
| `center_line_y` | Vertical center column → train *(simulated only)* |
| `concentric_square` | Inner square region → train *(simulated only)* |

### Checkpoints

One `state_dict` checkpoint is saved per epoch under `save_dir/sdn_epoch_{epoch}.pth`. Loss history is persisted as `save_dir/loss_history.pickle`.

---

## Design Decisions

**Full BPTT is required.** EDC, EDR, and EDP losses measure global energy decay properties. Gradients must flow from late reverberation back through the entire simulation loop to the junction parameters. Truncated BPTT is fundamentally incompatible with this objective. Operating at 16 kHz (vs. 48 kHz) is the correct lever for keeping sequence length and memory footprint manageable.

**System delay in meters.** The calibration artifact stores `δ` in meters, not samples, making it independent of the sampling rate used at inference time.

**Partial `load_state_dict` for cross-rate inference.** When evaluating a checkpoint at a different sample rate than training, delay-related buffer keys must be filtered before loading, allowing delay lines to be recomputed from geometry at the target rate.

---

## Dataset

Experiments use the **HOMULA-RIR** dataset:

> F. Miotello et al., "HOMULA-RIR: A Room Impulse Response Dataset for Teleconferencing and Spatial Audio Applications Acquired through Higher-Order Microphones and Uniform Linear Microphone Arrays," 2024 IEEE International Conference on Acoustics, Speech, and Signal Processing Workshops (ICASSPW), Seoul, Korea, Republic of, 2024, pp. 795-799, https://doi.org/10.1109/ICASSPW62465.2024.10626753

---

## Citation

```bibtex
@mastersthesis{magalini2025dsdn,
  author  = {Giorgio Magalini},
  title   = {Room Characterization and Novel-View Acoustics Synthesis
             With Differentiable Scattering Delay Networks},
  school  = {Politecnico di Milano},
  year    = {2026},
}
```