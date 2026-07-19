# Experiment History

## `bush4_stage2b_k3_cubic_envelope_full_5ep`

Date: 2026-07-17

### Purpose

Establish the stable full-frame Stage 2B baseline for bush4. Keep the staged,
motion-filled Gaussian mode shapes fixed and optimize only a separate
time-varying complex cubic envelope for each view and mode. This experiment
tests how much of the three stabilized videos can be reconstructed without
changing the solved 3D mode shapes or any static Gaussian parameters.

### Reusable paths

- Work directory: `/home/zs292/outputs_3dmode/bush4_stage2b_k3_cubic_envelope_full_5ep`
- Initialization checkpoint: `/home/zs292/outputs_3dmode/bush4_stage2b_k3_cubic_envelope_full_5ep/checkpoints/init.ckpt`
- Trained checkpoint: `/home/zs292/outputs_3dmode/bush4_stage2b_k3_cubic_envelope_full_5ep/checkpoints/last.ckpt`
- Static 3DGS checkpoint: `/home/zs292/outputs/bush4_sweep_static_3dgs_hq_v1/checkpoints/last.ckpt`
- K=3 modal manifest: `/home/zs292/outputs_modal/bush4/gaussian_modes_0p744_0p980_1p520_motion_fill_k8_d0p008/modal_modes_manifest.json`
- Dynamic RGB dataset: `/home/zs292/outputs_modal/bush4/dynamic_rgb_720_phase0`
- Modal frame map: `/home/zs292/outputs_modal/bush4/dynamic_rgb_720_phase0/modal_frame_map.json`
- View 1 config: `/home/zs292/outputs_modal/bush4/geometry/modal_surface_vggt/view1_config.json`
- View 2 config: `/home/zs292/outputs_modal/bush4/geometry/modal_surface_vggt/view2_config.json`
- View 3 config: `/home/zs292/outputs_modal/bush4/geometry/modal_surface_vggt/view3_config.json`

### Primary configuration

- Mode frequencies: 0.744, 0.980, and 1.520 Hz.
- Shape refinement: `fixed`; staged/motion-filled `phi` is not trained.
- Trainable parameter: `modal.params.envelope_knots` only.
- Envelope knot interval: 0.5 seconds.
- Envelope interpolation: cubic Hermite complex envelope.
- Gaussian geometry, appearance, opacity, scale, rotation, and background are frozen.
- Full frame set from all three views; no per-view frame cap.
- Epochs: 5; batch size: 4; data-loader workers: 1.
- Learning rate: envelope knots `1e-3`.
- Loss weights: RGB `1.0`, mask `1.0`, envelope magnitude `0.001`, envelope smoothness `1.0`, and envelope curvature `1.0`; depth, track, DCT, Gaussian, local-isometry, and `delta_phi` losses are disabled.

### Result and current conclusion

- This is the current stable fixed-`phi` baseline.
- The reconstructed motion is visibly more stable than the later `delta_phi`
  refinement experiments and does not show obvious Gaussian spreading artifacts.
- The trained envelopes visibly try to fit the original videos, but many motions
  still do not align with the source footage.
- The main observed limitation is the expressive span of only three spatial mode
  shapes, rather than clear evidence that the staged mode shapes require broad
  RGB-driven refinement.
- Reuse this checkpoint and the shared paths above when comparing future mode-set,
  activation, or reconstruction experiments.

## `bush4_stage2b_uniform_k10_0p3_2p0_cubic_envelope_full_5ep`

Date: 2026-07-18

### Purpose

Establish the current full-frame K=10 baseline for bush4. Ignore spectral peaks
when choosing the basis and instead solve ten uniformly spaced frequencies from
0.3 to 2.0 Hz. Keep all staged/motion-filled Gaussian mode shapes fixed and
train only a separate time-varying complex cubic harmonic envelope for each
view and mode. This experiment tests whether increasing the spatial basis from
K=3 to K=10 is sufficient to improve motion expressiveness before introducing
flow-derived coordinates, rigidity, or further phi refinement.

### Reusable paths

- Work directory: `/home/zs292/outputs_3dmode/bush4_stage2b_uniform_k10_0p3_2p0_cubic_envelope_full_5ep`
- Initialization checkpoint: `/home/zs292/outputs_3dmode/bush4_stage2b_uniform_k10_0p3_2p0_cubic_envelope_full_5ep/checkpoints/init.ckpt`
- Trained checkpoint: `/home/zs292/outputs_3dmode/bush4_stage2b_uniform_k10_0p3_2p0_cubic_envelope_full_5ep/checkpoints/last.ckpt`
- Analysis export: `/home/zs292/outputs_3dmode/bush4_stage2b_uniform_k10_0p3_2p0_cubic_envelope_full_5ep/analysis_export_20260718_200732`
- Static 3DGS checkpoint: `/home/zs292/outputs/bush4_sweep_static_3dgs_hq_v1/checkpoints/last.ckpt`
- K=10 modal manifest: `/home/zs292/outputs_modal/bush4/gaussian_modes_uniform_k10_0p3_2p0_motion_fill_k8_d0p008/modal_modes_manifest.json`
- Motion-fill graph: `/home/zs292/outputs_modal/bush4/gaussian_modes_uniform_k10_0p3_2p0_motion_fill_k8_d0p008/motion_fill/graph.npz`
- View 1 modal input: `/home/zs292/outputs_modal/bush4/view1/modal_analysis_uniform_k10_0p3_2p0.npz`
- View 2 modal input: `/home/zs292/outputs_modal/bush4/view2/modal_analysis_uniform_k10_0p3_2p0.npz`
- View 3 modal input: `/home/zs292/outputs_modal/bush4/view3/modal_analysis_uniform_k10_0p3_2p0.npz`
- Dynamic RGB dataset: `/home/zs292/outputs_modal/bush4/dynamic_rgb_720_phase0`
- Modal frame map: `/home/zs292/outputs_modal/bush4/dynamic_rgb_720_phase0/modal_frame_map.json`
- View 1 config: `/home/zs292/outputs_modal/bush4/geometry/modal_surface_vggt/view1_config.json`
- View 2 config: `/home/zs292/outputs_modal/bush4/geometry/modal_surface_vggt/view2_config.json`
- View 3 config: `/home/zs292/outputs_modal/bush4/geometry/modal_surface_vggt/view3_config.json`

### Primary configuration

- Frequencies: 0.300000, 0.488889, 0.677778, 0.866667, 1.055556,
  1.244444, 1.433333, 1.622222, 1.811111, and 2.000000 Hz.
- Gaussian count: 236,537 foreground Gaussians.
- 3D solver: staged Gaussian solve followed by joint KNN nullspace motion fill.
- Motion fill: K=8 and maximum scene-space distance 0.008.
- Pixel observations: stride 2, candidate K=4, preselect K=32, minimum
  render accumulation 0.05, and minimum contribution `1e-12`.
- Shape refinement: `fixed`; neither staged/motion-filled phi nor any Gaussian
  parameter is trained.
- Trainable parameter: cubic complex envelope knots only.
- Envelope knot interval: 0.5 seconds; interpolation is cubic Hermite complex.
- Full frames: view1 1,170; view2 1,013; view3 1,020.
- Epochs: 5; batch size: 4; data-loader workers: 1.
- Envelope learning rate: `1e-3`.
- Loss weights: RGB 1.0, mask 1.0, envelope magnitude 0.001, envelope
  smoothness 1.0, and envelope curvature 1.0. Depth, track, temporal RGB,
  Gaussian, local-isometry, and delta-phi losses are disabled.

### Quantitative result

- Overall RGB L1: 0.087172273 -> 0.083305007, delta -0.003867266.
- Overall PSNR: delta +0.465217 dB.
- Overall SSIM: delta +0.015827563.
- View1 RGB L1: 0.094341549 -> 0.091116546.
- View2 RGB L1: 0.082652218 -> 0.077120113.
- View3 RGB L1: 0.083521485 -> 0.080571139.
- All ten modes acquired nonzero envelopes. The two lowest-frequency modes,
  0.300000 and 0.488889 Hz, received the largest envelope RMS values overall.
- On 50,000 sampled short graph edges, absolute edge strain had p50 0.00953,
  p90 0.21683, and p99 2.24914; 24.68% exceeded 5% strain and 16.70%
  exceeded 10% strain. Neighbor motion directions were usually aligned
  (cosine p10 0.9771), but displacement magnitudes were not locally rigid.

### Result and current conclusion

- This is the current K=10 fixed-phi baseline for comparison with the new
  flow-derived per-frame coordinate experiment.
- Increasing from K=3 to K=10 clearly improved reconstruction metrics and gave
  the model more motion capacity. The videos looked generally plausible, but
  still did not match the source motion closely enough, especially in local
  amplitude and detailed movement.
- The remaining visible limitation is not only mode count: Gaussians belonging
  to the same structure can still deform with inconsistent magnitudes, so a
  future route needs an explicit local rigidity/structure constraint.
- The trained envelope checkpoint uses the removed
  `per_view_harmonic_envelope_v2` runtime and is intentionally incompatible
  with the new direct flow-coordinate code. Preserve its videos, metrics, and
  analysis export as the baseline; reuse the static checkpoint, K=10 manifest,
  motion-fill graph, frame map, dataset, and camera configs in new experiments.

## `bush4_flow_coordinates_k10_fixed_phi`

Date: 2026-07-18

### Purpose

Test the K=10 spatial basis without a learned harmonic envelope. Directly invert
every full-frame Farneback reference-flow field into one complex modal coordinate
per frame and mode, then render those fixed coordinates with the same fixed
staged/motion-filled Gaussian mode shapes. This separates temporal-coordinate
quality from RGB optimization and provides an expression-capacity experiment for
the current two-dimensional signal, solved three-dimensional basis, and local
Gaussian motion field.

### Reusable paths

- Work directory: `/home/zs292/outputs_3dmode/bush4_flow_coordinates_k10_fixed_phi`
- Initialization checkpoint: `/home/zs292/outputs_3dmode/bush4_flow_coordinates_k10_fixed_phi/checkpoints/init.ckpt`
- Final checkpoint: `/home/zs292/outputs_3dmode/bush4_flow_coordinates_k10_fixed_phi/checkpoints/last.ckpt`
- Reconstruction: `/home/zs292/outputs_3dmode/bush4_flow_coordinates_k10_fixed_phi/reconstruction_last`
- Coordinate directory: `/home/zs292/outputs_modal/bush4/flow_coordinates_uniform_k10_ridge1e4`
- Coordinate artifact: `/home/zs292/outputs_modal/bush4/flow_coordinates_uniform_k10_ridge1e4/modal_flow_coordinates.npz`
- Coordinate diagnostics: `/home/zs292/outputs_modal/bush4/flow_coordinates_uniform_k10_ridge1e4/diagnostics.json`
- Detailed coordinate diagnostics: `/home/zs292/outputs_modal/bush4/flow_coordinates_uniform_k10_ridge1e4/coordinate_diagnostics.npz`
- Static 3DGS checkpoint: `/home/zs292/outputs/bush4_sweep_static_3dgs_hq_v1/checkpoints/last.ckpt`
- K=10 modal manifest: `/home/zs292/outputs_modal/bush4/gaussian_modes_uniform_k10_0p3_2p0_motion_fill_k8_d0p008/modal_modes_manifest.json`
- Dynamic RGB dataset: `/home/zs292/outputs_modal/bush4/dynamic_rgb_720_phase0`
- Modal frame map: `/home/zs292/outputs_modal/bush4/dynamic_rgb_720_phase0/modal_frame_map.json`

### Primary configuration

- Frequencies/mode labels: the same ten uniformly spaced values from 0.3 to
  2.0 Hz as the preceding K=10 envelope baseline.
- Parameterization: `per_frame_flow_coordinates_v1`; frequency is a mode label
  and is not multiplied into the runtime coordinate.
- Coordinate solve: candidate-weighted projected fixed phi, reference-relative
  flow, ridge-relative weight `1e-4`, and per-view temporal mean-zero gauge.
- Coordinate shape: `[3203, 10]`, with full frames view1 1,170, view2 1,013,
  and view3 1,020.
- Staged/motion-filled phi, frequency labels, foreground/background Gaussians,
  and all appearance parameters remain fixed.
- No optimizer or training loop is created; trainable parameter count is zero.
  `init.ckpt` and `last.ckpt` contain identical model state.

### Qualitative result and current conclusion

- This route is visibly better than the preceding cubic-envelope direction and
  is the current fixed-phi K=10 baseline.
- Overall motion remains conservative. Two large flower clusters in the upper
  left are reconstructed particularly well, while most of the remaining bush
  has less varied motion than the source video.
- Rare frames show a small unnatural jerk. Some Gaussians on one physical
  branch move while nearby Gaussians remain nearly static, giving the branch a
  visibly stretched rather than approximately rigid motion.
- The conservative result has two leading explanations that remain entangled:
  the ten uniformly selected mode shapes may not span the bush's complex motion,
  and the Farneback-derived two-dimensional signal may not contain sufficiently
  accurate motion for either the FFT mode shapes or the per-frame coordinate
  inversion.
- The within-structure stretching is a separate spatial-field issue: the fixed
  staged/motion-filled phi can assign inconsistent displacement magnitudes to
  neighboring Gaussians even when the shared coordinates are reasonable.
- The next priority is a better modal representation: first distinguish
  two-dimensional signal quality from basis-span limitations, then improve mode
  count/selection or replace the 2D signal estimator. Shared phi refinement and
  local rigidity should follow only after that diagnosis; a later stage may
  integrate the complete modal model into dynamic 3DGS optimization instead of
  remaining attached to a frozen static Gaussian reconstruction.

## `dense_exact_dft_greedy_20260719_002548`

Date: 2026-07-19

### Purpose

Measure how much of the K=10 uniform-frequency basis limitation comes from the
frequency choices themselves before paying for any additional 3D solves. Build
an in-sample exact-DFT dictionary on a shared 0.2--2.2 Hz grid with 0.025 Hz
spacing, treat each real/imaginary pair as one inseparable complex mode, and use
equal-view greedy residual reduction to report nested K=5/10/15/20 direct-2D
capacity on the existing pixel-candidate ROI.

### Reusable paths

- Output directory: `/home/zs292/outputs_modal/bush4/frequency_selection/dense_exact_dft_greedy_20260719_002548`
- Summary: `/home/zs292/outputs_modal/bush4/frequency_selection/dense_exact_dft_greedy_20260719_002548/frequency_selection_summary.json`
- Detailed diagnostics: `/home/zs292/outputs_modal/bush4/frequency_selection/dense_exact_dft_greedy_20260719_002548/frequency_selection_diagnostics.npz`
- Capacity plot: `/home/zs292/outputs_modal/bush4/frequency_selection/dense_exact_dft_greedy_20260719_002548/frequency_selection_curves.png`
- Export-compatible shortlists: `selected_frequencies_k5.json`,
  `selected_frequencies_k10.json`, `selected_frequencies_k15.json`, and
  `selected_frequencies_k20.json` in the same output directory.
- Source K=10 manifest: `/home/zs292/outputs_modal/bush4/gaussian_modes_uniform_k10_0p3_2p0_motion_fill_k8_d0p008/modal_modes_manifest.json`
- Source frame map: `/home/zs292/outputs_modal/bush4/dynamic_rgb_720_phase0/modal_frame_map.json`

### Primary configuration and result

- Candidate grid: 81 exact-DFT frequencies from 0.2 through 2.2 Hz; no amplitude
  clamp; full time series used both to form and evaluate each spatial mode.
- Candidate pixels: view1 34,672; view2 24,961; view3 20,563.
- Pooled R2: K=5 0.719651, K=10 0.829752, K=15 0.885500, and K=20
  0.915107. The previous uniform K=10 direct-2D baseline was 0.726825.
- Equal-view macro R2: K=5 0.700382, K=10 0.818885, K=15 0.880356,
  and K=20 0.913175. This is the primary capacity score because view1 carries
  about 66.9% of the pooled flow energy, versus 24.6% for view2 and 8.5% for
  view3.
- Worst-view R2: K=5 0.676569, K=10 0.793683, K=15 0.860371, and
  K=20 0.900117. Every view retained the full real rank 2K.
- The K=15 greedy prefix was 0.2, 0.225, 1.525, 0.425, 0.85, 1.25,
  0.525, 0.275, 1.65, 1.4, 1.8, 1.025, 0.3, 1.725, and 1.1 Hz in
  selection order.
- The apparent worst 4x4 regions carried very little motion energy: at K=20 the
  minimum-R2 regions accounted for approximately 0.34% of view1 energy, 0.25%
  of view2 energy, and 0.0094% of view3 energy.

### Current conclusion

- Frequency selection matters substantially: greedy K=5 nearly matches the old
  uniform K=10, while greedy K=10 improves pooled R2 by about 0.103 and reduces
  the old unexplained flow energy by about 37.7%.
- K=15 is the current capacity knee; K=10 to K=15 adds about 0.056 pooled R2,
  whereas K=15 to K=20 adds about 0.030.
- These are optimistic in-sample capacity values: the same complete sequences
  were used to select and score the frequencies. They do not yet demonstrate
  held-out temporal generalization or identify 15 physical natural modes.
- The shortlist is not yet a physical natural-frequency list. The first two
  atoms are adjacent 0.2/0.225 Hz frequencies at the lower search boundary and
  are already strongly correlated; later 0.275/0.3/0.375/0.525/0.55 Hz atoms
  also become highly redundant. Greedy is using nearby whole-clip DFT snapshots
  to span slow, broadband, or nonstationary motion.
- The early selection of 1.525 Hz is consistent with the previously useful
  1.52 Hz mode, but it is especially strong in view2 rather than uniformly
  strong in all three videos.
- Do not run a full K=20 3D solve from this list yet. The next experiment should
  first test whether the dominant low-frequency atom follows the lower search
  boundary, then use Welch/FDD or SPOD-style window stability, frequency-band
  grouping, and blocked temporal validation to distinguish repeatable modes
  from drift/noise. Only then run a staged no-fill anchor/partial pilot before
  full motion fill.
