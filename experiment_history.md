# Experiment History

## `corn_static_orientation_and_resolution_diagnosis`

Date: 2026-07-26

### Purpose

Diagnose the fixed-camera orientation and memory issue before rebuilding the
formal corn joint-COLMAP geometry.

### Paths and observations

- Current fixed-camera images:
  `/home/zs292/datasets/custom/images/corn1` and
  `/home/zs292/datasets/custom/images/corn2`.
- Preserved full-resolution images:
  `/home/zs292/datasets/custom/images/corn1_fullres_backup` and
  `/home/zs292/datasets/custom/images/corn2_fullres_backup`.
- `corn1` contains 1,789 current and 1,789 backup frames; `corn2` contains
  1,763 current and 1,763 backup frames. No current frame is empty.
- Current images are `1080x1920`; backups are `2160x3840`.

### Result

- The fixed-camera source videos are portrait-encoded, while their scene
  content needs a manual 90-degree counterclockwise rotation to become upright
  landscape imagery. This is not an FFmpeg autorotation correction.
- The accepted replacement transform is counterclockwise rotation followed by
  a `0.5` uniform scale, producing `1920x1080` static references and complete
  fixed-camera sequences.
- The portrait 1080 sweep is already correct and must not be rotated or scaled.
- The successful `joint_colmap_v1` geometry is retained only as the previous
  orientation baseline and will be superseded by a separately versioned
  landscape-reference reconstruction after cluster validation.

## `corn_joint_colmap_v1_formal_success`

Date: 2026-07-26

### Purpose

Build the formal video-only joint-COLMAP geometry for corn from a complete
15 FPS portrait sweep and one 3.0-second landscape reference from each of the
two fixed-camera videos.

### Paths and configuration

- Output directory:
  `/home/zs292/data_formal/corn_2view_v1/shared/preprocessing/joint_colmap_v1`
- Sweep video: `/home/zs292/datasets/custom/videos/corn_sweep.MOV`
- Static videos: `/home/zs292/datasets/custom/videos/corn1.mov` and
  `/home/zs292/datasets/custom/videos/corn2.mov`
- Reference timestamps: 3.0 seconds for both fixed views.
- Sweep sampling rate: 15 FPS.
- Camera model: `SIMPLE_RADIAL`.
- Camera grouping: one camera for the portrait sweep and one shared camera for
  both landscape references.

### Result

- The sole complete model registered all 280 sweep frames and both static
  references, for 282/282 registered images and a sweep registration ratio of
  1.0.
- The reconstruction contains 125,061 sparse points, 1,787,878 observations,
  a mean track length of 14.296, and a mean reprojection error of 0.642464 px.
- COLMAP reported two cameras with the intended group IDs: sweep camera 2 and
  shared static-reference camera 1.
- The packaged sweep dataset, normalized scene transform,
  `scene_norm_dict.pth`, registered-camera report, and fixed-reference camera
  manifest were all written successfully.
- No VGGT alignment is required. The next required input for static 3DGS is a
  foreground/background mask for every packaged sweep image.

## `corn_joint_colmap_v1_shared_reference_directory_failure`

Date: 2026-07-26

### Purpose

Run the first video-only joint-COLMAP preparation for corn using a 15 FPS
portrait sweep and one 3.0-second landscape reference from each fixed view.

### Paths and configuration

- Output directory: `/home/zs292/outputs_modal/corn/geometry/joint_colmap_v1`
- Sweep video: `/home/zs292/datasets/custom/videos/corn_sweep.MOV`
- Static videos: `/home/zs292/datasets/custom/videos/corn1.mov` and
  `/home/zs292/datasets/custom/videos/corn2.mov`
- Reference timestamps: 3.0 seconds for both fixed views.
- Sweep sampling rate: 15 FPS.

### Result

- The run failed before COLMAP feature extraction while exporting the second
  reference image.
- Root cause: both static references intentionally share
  `colmap_workspace/images/static_refs`, but the directory creation rejected
  the folder already created for the first reference.
- The extraction helper now permits the shared directory to exist while each
  reference image still uses FFmpeg no-overwrite behavior.
- `joint_colmap_v1` is a partial failed output and should not be reused; rerun
  into a new versioned output directory.

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

## `bush4_flow_coordinates_greedy_k15_quality_20260719_v1_fixed_phi`

Date: 2026-07-19

### Purpose

Test reconstruction quality with the quality-first K=15 prefix selected by the
dense exact-DFT greedy experiment. This experiment deliberately accepts that
nearby low-frequency Fourier atoms may span broadband or nonstationary slow
motion rather than represent fifteen independent physical natural modes. Its
purpose is to establish whether the richer fixed 3D basis is already strong
enough to serve as the initialization for subsequent joint dynamic-3DGS
optimization.

### Reusable paths

- 3D modal output: `/home/zs292/outputs_modal/bush4/gaussian_modes_greedy_k15_quality_20260719_v1_motion_fill_k8_d0p008`
- Modal manifest: `/home/zs292/outputs_modal/bush4/gaussian_modes_greedy_k15_quality_20260719_v1_motion_fill_k8_d0p008/modal_modes_manifest.json`
- Flow-coordinate directory: `/home/zs292/outputs_modal/bush4/flow_coordinates_greedy_k15_quality_20260719_v1_ridge1e4`
- Flow-coordinate artifact: `/home/zs292/outputs_modal/bush4/flow_coordinates_greedy_k15_quality_20260719_v1_ridge1e4/modal_flow_coordinates.npz`
- Fixed-coordinate run: `/home/zs292/outputs_3dmode/bush4_flow_coordinates_greedy_k15_quality_20260719_v1_fixed_phi`
- Final checkpoint: `/home/zs292/outputs_3dmode/bush4_flow_coordinates_greedy_k15_quality_20260719_v1_fixed_phi/checkpoints/last.ckpt`
- Reconstruction: `/home/zs292/outputs_3dmode/bush4_flow_coordinates_greedy_k15_quality_20260719_v1_fixed_phi/reconstruction_last`
- Clean static initialization checkpoint: `/home/zs292/outputs_modal/bush4/static_checkpoints/bush4_static_for_modal_solver.ckpt`

### Primary configuration

- Frequencies in greedy prefix order: 0.2, 0.225, 1.525, 0.425, 0.85, 1.25,
  0.525, 0.275, 1.65, 1.4, 1.8, 1.025, 0.3, 1.725, and 1.1 Hz.
- Each 3D mode uses the staged multi-view Gaussian solve followed by distance-KNN
  motion fill with `k=8` and maximum edge distance `0.008` scene units.
- Per-frame complex coordinates are inverted from the three full Farneback flow
  caches with relative ridge weight `1e-4`.
- The 3D modal fields, coordinates, static Gaussian geometry, and appearance are
  all fixed; no RGB optimizer or training loop is used.

### Qualitative result and current conclusion

- The reconstruction quality is considered sufficient to become the current
  formal baseline and initialization for joint optimization.
- The richer K=15 representation is visibly preferable to the earlier limited
  bases, so basis capacity is no longer the immediate blocking issue for
  beginning trainable-phi and dynamic-Gaussian experiments.
- Motion is generally convincing, but rare frames still contain very slight
  temporal twitching. This should be addressed by a weak coordinate data prior
  plus temporal smoothing or joint coordinate refinement, without returning to
  the earlier restrictive cubic-envelope parameterization.
- The next stage should jointly refine modal coordinates and shared phi under an
  explicit basis gauge and local structural constraints. Gaussian geometry and
  appearance should be opened only in controlled stages so they cannot absorb
  motion errors before the deformation model is stable.

### Known static-scene limitation

- The inherited static scene is adequate for the current foreground/modal
  baseline, but its background is visibly lower quality than a vanilla static
  3DGS reconstruction. This is a known limitation of the base checkpoint rather
  than evidence that the K=15 modal representation is incorrect.
- The source static checkpoint is
  `/home/zs292/outputs/bush4_sweep_static_3dgs_hq_v1/checkpoints/last.ckpt`;
  `/home/zs292/outputs_modal/bush4/static_checkpoints/bush4_static_for_modal_solver.ckpt`
  is only a solver-compatible copy of the same static field and does not improve
  its background.
- The current representation stores one view-independent RGB value per Gaussian
  rather than the spherical-harmonic appearance used by vanilla 3DGS. The static
  foreground/background split also classifies sparse COLMAP points by mask votes
  without point-track visibility or occlusion testing; ties and unobserved points
  are assigned to foreground. Both groups are then initialized independently,
  leaving the spatially broader background comparatively sparse and smooth.
- Additional inherited differences include a five-times larger background
  opacity learning rate, scale-variance regularization, and adaptive density
  control ending at global step 4,000 even though this checkpoint was optimized
  for roughly 48,000 steps. These factors may further disadvantage distant or
  low-contrast background structure.
- Until the static representation is replaced or improved, joint modal training
  should initially keep the background frozen and prevent background RGB/flow
  residuals from being interpreted as evidence for changing modal coordinates or
  phi. Foreground quality is considered sufficient to continue the current line
  of experiments.

## `bush4_physics_coordinates_k15_zeta0p05_force0p1_fixed_phi_20260719_200201`

Date: 2026-07-19

### Purpose

Test whether the free per-frame K15 flow coordinates can be made more physically
meaningful without training phi or any Gaussian parameter. Each view/mode complex
coordinate was post-fit with the fixed assigned frequency, damping ratio 0.05,
latent-force weight 0.1, and zero force-difference weight. The resulting coordinates
were materialized as a fixed checkpoint and rendered with the same staged K15 phi.

### Reusable paths

- Input coordinates: `/home/zs292/outputs_modal/bush4/flow_coordinates_greedy_k15_quality_20260719_v1_ridge1e4/modal_flow_coordinates.npz`
- Physics coordinate directory: `/home/zs292/outputs_modal/bush4/physics_coordinates_greedy_k15_zeta0p05_force0p1_20260719_200201`
- Physics coordinate artifact: `/home/zs292/outputs_modal/bush4/physics_coordinates_greedy_k15_zeta0p05_force0p1_20260719_200201/modal_flow_coordinates.npz`
- Fixed-phi run: `/home/zs292/outputs_3dmode/bush4_physics_coordinates_k15_zeta0p05_force0p1_fixed_phi_20260719_200201`
- Reconstruction: `/home/zs292/outputs_3dmode/bush4_physics_coordinates_k15_zeta0p05_force0p1_fixed_phi_20260719_200201/reconstruction_last`

### Quantitative result

- Overall candidate-flow R2 changed from 0.841542 to 0.809967, a decrease of
  0.031575. View losses were 0.025441, 0.040957, and 0.052719 respectively.
- Strong-motion flow R2 was retained better: view1 0.885702 to 0.867791, view2
  0.893994 to 0.861004, and view3 0.835016 to 0.811721.
- Mean normalized latent-force RMS fell from 16.9498 to 0.7469, while mean
  assigned-frequency energy rose from 0.2344 to 0.2848.
- Mean coordinate RMS, p90, and p99 retention were approximately 0.8920, 0.8983,
  and 0.8734; coordinate fidelity NRMSE was 0.2370.
- RGB reconstruction was effectively unchanged: overall L1 improved by only
  0.0000151, PSNR changed by -0.00275 dB, and SSIM improved by 0.000281.

### Current conclusion

- The post-fit successfully removes most high-force/nonphysical coordinate content
  without collapsing strong amplitudes. Although weight 0.1 reduces overall flow R2
  by 3.16 points and disproportionately affects views 2 and 3, direct video review
  shows substantially fewer temporal twitches and no new dominant artifact. The
  visual stability gain is accepted over the nominal two-point flow-R2 budget.
- The low input assigned-band energy confirms that the original free coordinates
  rely heavily on cross-frequency content. Raising the ratio only to 0.285 despite
  the large force reduction also shows that the current phi-frequency association
  cannot yet explain all observed motion as lightly forced physical modes.
- This run is now the temporal-stability K15 baseline for subsequent experiments.
  The original fixed-coordinate K15 run remains the higher-flow-fidelity reference.
  Remaining visible errors are mainly insufficient motion expressivity and imperfect
  fixed mode shapes, rather than the previously dominant coordinate twitching.
- A weaker forcing weight such as 0.01 remains an optional ablation, not the next
  required step. Future phi or representation experiments should compare against
  both this stable baseline and the original free-coordinate reconstruction.
## `mode4_anchor_graph_exp`

Date: 2026-07-20

### Purpose

Build and inspect the diagnostic observed-anchor structure graph for only the
fifth solved bush4 mode, without motion fill, refinement, or Gaussian training.

### Reusable paths

- Output directory: `/home/zs292/outputs_modal/bush4/mode4_anchor_graph_exp`
- Mode index: `4` (`0.850 Hz`)

### Primary configuration and result

- Pixel observation sampling: stride `2`, candidate K `4`, preselect K `32`,
  rendered-accumulation minimum `0.05`, contribution minimum `1e-12`, and one
  mask-erosion iteration.
- Anchor graph: mutual KNN with at most `8` neighbors and scene-distance cutoff
  `0.008`; motion fill was not requested.
- Observation construction reported 236,537 foreground Gaussians, 291,962 rows,
  and per-view row counts of 128,722, 88,409, and 74,831.
- The saved graph exposed 6,703 retained edges in the Viewer controls.
- The main Viewer could load the graph metadata but could not render or update
  it because its pinned `viser==0.2.1` lacks `SceneApi.add_line_segments`; graph
  inspection is therefore being moved to an isolated modern-Viser viewer, with
  no need to rerun this solve.

### Observation-coverage replay

- Diagnostic artifact: `/home/zs292/outputs_modal/bush4/mode4_anchor_graph_exp/observation_coverage_mode4.npz`
- The static replay exactly reproduced the saved K=4 per-Gaussian view counts,
  sample counts, and per-view observation totals before emitting diagnostics.
- Multi-view selected Gaussian counts for K=4, 8, 12, 16, and 32 were 23,106,
  34,710, 40,305, 42,719, and 43,931, respectively.
- Unobserved counts over the same sweep were 169,298, 149,998, 141,542,
  137,661, and 135,446.
- The 43,931 Gaussians with positive-contribution candidates in at least two
  views are independent of selected K. K=4 retained 23,106 (52.6%) of them and
  lost 20,825 (47.4%) specifically through top-K competition; K=8, 12, and 16
  retained 79.0%, 91.7%, and 97.2%, respectively.
- Separately, 160,032 Gaussians were preselected in fewer than two views, and
  32,574 were preselected in at least two views but retained positive
  contribution/camera depth in fewer than two views. These counts do not change
  with selected K.
- Conclusion: top-4 competition is a major confirmed source of anchor sparsity,
  but increasing K alone cannot yield dense coverage. K=12 captures most of the
  available multi-view candidates with substantially diminishing gains beyond
  it; the larger remaining bottleneck lies before top-K selection, especially
  insufficient multi-view preselection coverage.

### Selected-multiview to anchor waterfall

- All 23,106 K=4 selected-multiview Gaussians retained at least two
  alpha-identifiable views.
- Weighted projection-Jacobian rank rejected 1,340 Gaussians, leaving 21,766;
  the condition-limit and finite-residual checks rejected none of these.
- The saved normalized residual threshold was the configured cap `0.1`. It
  rejected 17,798 of the remaining Gaussians, leaving only 3,968 final anchors.
- The anchor graph retained all 3,968 nodes in its artifact: 3,059 received at
  least one accepted structure edge, 909 remained isolated, and 6,703 edges
  were accepted.
- Revised anchor-specific conclusion: raw K=4 multi-view coverage is already
  spatially rich, and observation rank is rarely the limiting condition. The
  dominant reduction from selected-multiview points to anchors is inconsistent
  multi-pixel/multi-view modal observations failing the single-3D-phi residual
  test, not graph construction. Increasing candidate K may add coverage but
  does not by itself address this consistency bottleneck.

### Exact anchor-residual decomposition

- Diagnostic artifact: `/home/zs292/outputs_modal/bush4/mode4_anchor_graph_exp/anchor_residual_decomposition_mode4.npz`
- Exact replay reproduced the saved `point_precompletion_residual`, the 21,766
  residual candidates, 17,798 residual rejects, and 3,968 accepted anchors.
- Of the residual-rejected Gaussians, 15,607 (87.69%) were cross-view dominated,
  1,047 (5.88%) were within-view dominated, 1,049 (5.89%) were mixed, and only
  95 (0.53%) had robustly low modal energy.
- Cross-view disagreement contributed 4.48587e6 weighted SSE (94.52%) versus
  260,194 (5.48%) from within-view pixel-mode dispersion. Median rejected-point
  total, within-view, and cross-view normalized residuals were 0.287349,
  0.0280773, and 0.264511, respectively; the median within-SSE fraction was
  0.00903953.
- Per-view rejected cross SSE was 1.52938e6, 1.88883e6, and 1.06765e6 for views
  1, 2, and 3, but these raw totals are not normalized for each view's signal
  energy or observation mass and therefore do not by themselves identify a bad
  camera.
- Conclusion: the anchor bottleneck is overwhelmingly a cross-view compatibility
  failure under the current shared 3D phi, synchronized alpha, and projection
  model. Broad within-view denoising alone is unlikely to recover most rejected
  anchors; the next diagnosis should separate a global/view-level synchronization
  error from spatially local correspondence, occlusion, or same-frequency mode
  mixing errors.

## `mode4_anchor_graph_residual0p5`

- Output directory: `/home/zs292/outputs_modal/bush4/mode4_anchor_graph_residual0p5`
- The experiment solved only mode index 4 and relaxed the staged anchor residual
  cap from `0.1` to `0.5`; motion fill and learned refinement were not enabled.
- The diagnostic structure graph retained the existing mutual-KNN settings of at
  most 8 neighbors and a `0.008` scene-unit distance cutoff, with the same Lab
  color and rendered-depth checks.
- The standalone Viewer reported 45,053 accepted graph edges. Visual inspection
  showed substantially denser, locally coherent flower-head structures than the
  residual-0.1 graph, while still leaving isolated points and incomplete regions.
- Working conclusion: the color-depth graph construction has useful structural
  precision on a much broader candidate set; the dominant sparsity came from
  defining graph nodes only after the strict first-pass residual filter. This
  motivates the next diagnostic graph over every positive-weight observed
  Gaussian before changing the solver.

## `mode4_rigid_components_full_k8_d0p008`

- Output directory: `/home/zs292/outputs_modal/bush4/mode4_rigid_components_full_k8_d0p008`
- The mode-4 observed-graph rigid-component pipeline completed successfully and
  wrote `modal_modes_manifest.json` with solver exit code 0.
- The reused source observation contained 236,537 foreground Gaussians and
  291,962 rows: 128,722 from view1, 88,409 from view2, and 74,831 from view3;
  169,298 Gaussians had no observation row.
- This is the first completed rigid-component solve after separating the strict
  complex128 first-order rigidity check from the quantization-accounted
  complex64 persistence check. Visual motion and component/fill diagnostics
  remain to be inspected in the standalone Viewer.
- Standalone Viewer inspection found that most visibly flying components were
  highlighted by the default single-view anomaly diagnostic, supporting
  insufficient multi-view support as the dominant failure mode.
- The low singular-ratio diagnostic predominantly selected very small
  components containing only one or two accepted graph edges. These are good
  candidates for removing from the trusted rigid-seed set rather than deleting
  their Gaussians or changing canonical indices.
- The P99 component-motion-RMS diagnostic selected only a small subset of the
  visibly incorrect components, but every selected component was flying. Motion
  RMS is therefore a high-precision, low-recall outcome diagnostic and should
  not be the primary rejection rule.
- Rebuilt the mode-4 observed graph as
  `/home/zs292/outputs_modal/bush4/mode4_observed_graph_exp/mode_004_0p85hz_observed_structure_graph_v2.npz`
  with the four-node/three-edge graph-pruning contract, then ran the no-fill
  rigid baseline in
  `/home/zs292/outputs_modal/bush4/mode4_rigid_components_no_motion_fill_profile`.
  The full pipeline took 25.177 s: setup 6.609 s, alpha synchronization 5.493 s,
  rigid component solve 0.868 s, and visualization/output 11.706 s. This confirms
  that the previous half-hour run is dominated by motion fill rather than the
  rigid solve.
- Viewer inspection confirmed that visibly stretched retained components align
  with high finite-amplitude drift. A threshold sweep rejected 126 components,
  1,063 nodes, and 1,895 edges at maximum drift 2.0, versus 201/3,923/8,484 at
  1.0 and 277/12,942/30,573 at 0.5. The next solver revision therefore uses 2.0
  as the default finite-drift trusted-seed cap to remove extreme deformation
  while affecting about 0.45% of the 236,537 foreground Gaussians.
- The earlier pointwise motion-fill path was reported to finish in roughly
  10--15 minutes, while the grouped single-view-component run associated with
  `/home/zs292/outputs_modal/bush4/mode4_rigid_components_trusted_v2_k8_d0p008`
  remained unfinished after more than 30 minutes. Since the no-fill baseline
  finishes in 25.177 s, the next revision targets grouped-only repeated point
  scans first and adds live connectivity/assembly/LSMR/reconstruction timings
  so the next run can distinguish Python preparation overhead from LSMR cost.
- The optimized joint grouped run in
  `/home/zs292/outputs_modal/bush4/mode4_grouped_motion_fill_optimized_profile_retry`
  completed in 2,374.810 s. Motion fill consumed 2,351.347 s (99.7% of the
  solve/fill time), with real and imaginary LSMR taking 1,169.362 s and
  1,168.312 s respectively; preparation, connectivity, assembly, layout,
  reconstruction, and validation together took under five seconds. Viewer
  inspection found coherent, non-splitting component motion and generally good
  visual quality, while confirming that the joint route obtains rejected
  single-view component twists from KNN smoothness rather than retaining their
  observable single-view motion.
- The first component-only partial-fill run in
  `/home/zs292/outputs_modal/bush4/mode4_single_view_partial_component_fill_r1e2_ray0p8`
  prepared 249 groups containing 6,482 points. Its reduced graph had 441
  eligible edges, 262 active edges, 49 trusted-connected groups, and a
  786-by-70 sparse system with 1,356 nonzeros. Both real and imaginary LSMR
  reached SciPy's default 70-iteration limit in milliseconds with stop code 7,
  so no artifact was published; this identifies the default column-count
  iteration cap, rather than runtime or matrix size, as the immediate failure.
- After raising the component-only LSMR budget, the same partial-fill experiment
  completed its solve and wrote the modal manifest and artifacts in 28.784 s of
  pipeline time. Per-mode fill took 0.253 s, including 0.005 s real and 0.004 s
  imaginary LSMR, versus 2,342.102 s for the joint grouped baseline. The process
  then exited while printing the already-written timing profile because the
  summary formatter still required joint-only layout and reconstruction keys;
  this is a profiling-output defect rather than a solve or artifact failure.
- The standalone Viewer initially rejected the completed partial artifact at
  component 515 because it compared the independently persisted complex64 point
  field and complex64 rigid twist with a fixed allclose tolerance. The partial
  solver constructs the field exactly from the twist in complex128 before those
  arrays are rounded separately, so large cancelling translation/rotation terms
  require an operand-scaled real/imaginary quantization bound during artifact
  validation rather than a tolerance based only on the final motion amplitude.
- Viewer inspection then showed that most red single-view anomaly components
  remained static. The component-only run had selected 249 groups but connected
  only 49 to trusted anchors; the implementation copied observable motion only
  for those connected groups and explicitly zeroed every other final twist.
  This contradicted the intended partial model: unconnected selected components
  should retain their stable non-ray observable twist with zero weak-direction
  correction, while remaining marked incomplete for KNN completion.
- Following inspection of
  `/home/zs292/outputs_modal/bush4/mode4_single_view_partial_component_fill_r1e2_ray0p8`,
  the next component-only revision rejects excessive candidate motion after
  reconstruction but before publication. It zeros a whole single-view component
  when finite-amplitude edge drift exceeds `2.0` or point-motion RMS divided by
  component radius exceeds `2.0`, removes rejected groups from completion and
  future anchor connectivity, and retains the pre-zero metrics and separate
  rejection masks for Viewer diagnosis.
- Viewer comparison of the radius-gated and ungated single-view partial runs
  showed that some components with plausible motion in their source view became
  inconsistent with adjacent trusted structures from side views. Inspection
  identified raw component-level view presence as too permissive: one sparsely
  observed secondary region could promote an otherwise single-view component.
  The next run directory is
  `/home/zs292/outputs_modal/bush4/mode4_single_view_partial_view_support_1over3_no_radius_gate`;
  it requires every supporting view to observe at least one third as many
  distinct component nodes as the dominant view, keeps the normalized-motion
  gate disabled for isolation, and sends downgraded effective-single-view
  components through the existing partial KNN fill.
- A secondary/dominant distinct-node coverage sweep over 438 raw multiview
  components found ratios p10/p25/p50 of 0.1595/0.2893/0.5921. Thresholds
  0.05, 0.10, 0.15, 0.20, 0.25, and 1/3 downgraded 8, 19, 43, 59, 76, and
  117 components respectively. Viewer inspection of
  `/home/zs292/outputs_modal/bush4/mode4_single_view_partial_view_support_0p15_no_radius_gate`
  found the 0.15 version visually reasonable, so subsequent component-only
  experiments retain 0.15 while increasing KNN influence separately.
- The next full-fill comparison uses
  `/home/zs292/outputs_modal/bush4/mode4_sequential_partial_then_pointwise_independent_profile`.
  It removes the earlier simultaneous grouped-component/point solve: trusted
  components and finite-safe trusted-connected single-view completions are
  frozen after the fast component stage, while every remaining Gaussian is
  reset and solved as an independent 3D point variable. The run profile now
  separates component-stage time from pointwise connectivity, assembly,
  real/imaginary LSMR, and reconstruction time.
- The next bounded-propagation comparison uses
  `/home/zs292/outputs_modal/bush4/mode4_sequential_partial_then_pointwise_hop8_profile`.
  It retains the sequential partial-component anchors but restricts the
  independent-Gaussian LSMR to points within eight full-graph KNN hops of a
  fixed anchor. Farther connected targets remain zero and incomplete, while
  their original hop distances and the total truncated count remain diagnostic.
- The tolerance comparison uses
  `/home/zs292/outputs_modal/bush4/mode4_sequential_partial_then_pointwise_hop8_tol1e6_profile`.
  It keeps the eight-hop sequential graph and changes both real and imaginary
  component/pointwise LSMR stopping tolerances from `1e-10` to `1e-6`; the run
  should be compared by iteration count, LSMR wall time, and final Viewer motion.
- The corrected component-anchor comparison uses
  `/home/zs292/outputs_modal/bush4/mode4_finite_safe_single_view_anchors_hop8_tol1e6_profile`.
  All single-view components that pass finite drift now keep their component
  motion and seed downstream Gaussian interpolation; trusted-KNN connectivity
  only supplies weak/ray coefficients. The eight-hop cutoff is measured from
  this expanded anchor set and applies only to remaining independent Gaussians.

## `bush4_rigid_components_k20_physics_zeta0p05_force0p1_fixed_phi_reconstruction`

Date: 2026-07-22

### Purpose

Record the latest accepted bush4 integration baseline before unifying the two
development branches. The run uses the `rgbOptTest` rigid-component solver and
sequential motion fill to produce fixed K20 Gaussian mode shapes, then consumes
that manifest with the `modal718-rigid-fill-compat` flow-coordinate, physics
post-fit, checkpoint materialization, and reconstruction pipeline.

### Reusable path

- Reconstruction and comparison directory: `/home/zs292/outputs_3dmode/compare/bush4_rigid_components_k20_physics_zeta0p05_force0p1_fixed_phi_reconstruction`

### Primary configuration and role

- Mode count: K20.
- Spatial field: rigid-component solve followed by sequential motion fill;
  filled finite-safe components remain fixed anchors.
- Physics coordinate post-fit: damping ratio `0.05`, forcing weight `0.1`.
- Gaussian mode shapes remain fixed during coordinate materialization and
  rendering.
- This is the regression baseline for the unified rigid/physics code path. Its
  Linux artifacts and comparison videos were not regenerated in the local
  Windows merge workspace.

## `rigid_rgbdepth_v3_sequential_k60_bounded_complex_v1`

Date: 2026-07-23

### Failed run and follow-up

- Output directory:
  `/home/zs292/data_formal/bush4_3view_v1/modal_fields/greedy_0p2_4p0_step0p025_k60prefix_v1/rigid_rgbdepth_v3_sequential_k60_bounded_complex_v1`
- The bounded-complex K60 rerun reached mode 11 at 1.025 Hz, excluded view2
  during alpha synchronization, and then stopped in single-view partial
  component preparation because a component member observed only in the
  excluded view had no usable viewing ray.
- The follow-up keeps alpha exclusion unchanged and restricts ray-direction
  classification plus radial/tangent diagnostics to component members with
  usable observations. Other members retain the shared rigid twist and
  downstream KNN participation.

### Interrupted K60 continuation

Date: 2026-07-24

- The rerun in the same output directory reached 39 completed mode latents
  after approximately 6 hours 53 minutes, while the Slurm allocation retained
  its original 8-hour limit and could not be extended by the submitting user.
- Profiling from the live log showed bounded-complex alpha synchronization,
  rather than rigid solving or the roughly one-minute sequential motion fill,
  as the dominant per-mode cost.
- The solver and `run.sbatch` now use explicit rigid-component resume mode:
  completed latent/diagnostic pairs are validated and reused, an interrupted
  mode without its final latent is recomputed, and the remaining modes continue
  in the same directory before one complete K60 manifest is written.
- Slurm job `317439` resumed through mode 41 and stopped on
  `mode_042_0p5hz`: all three views were structurally reference-connected with
  strong overlap, rank ratio `0.999868`, information ratio `0.139661`, condition
  `1.000132`, and no active gain bound, but both non-reference views were
  excluded as `optimizer_failure` because the bounded-complex frozen-weight
  Huber IRLS did not reach its strict 12-round fixed point. The resulting
  reference-only alpha left zero multiview rigid components and therefore no
  trusted seed for motion fill.
- The follow-up replaces the capped IRLS with a single exact block-Huber
  residual transformation while retaining the batched profiled point solve.
  Modes 0--41 remain reusable; mode 42 has no final latent and will be
  recomputed by the next resumed job.
- The exact block-Huber resumed run subsequently completed through mode 46
  and stopped on `mode_047_2p35hz`. Its final two-view candidate retained
  strong overlap, rank ratio `0.973404`, information ratio `0.157430`,
  condition `1.027323`, and no active gain bound, but SciPy stopped at
  `max_nfev=100` before convergence and both non-reference views were
  ultimately excluded as `optimizer_failure`.
- The follow-up increases only the bounded-complex exact block-Huber
  evaluation budget to 500. Modes 0--46 remain reusable; mode 47 has no final
  latent and will be recomputed by the next resumed job.
- The resumed bounded-complex solve subsequently completed all 60 modes and
  wrote its final manifest in the same output directory.
- The first per-frame flow-coordinate attempt stopped before creating its
  output because the downstream manifest loader still required every
  mode/view alpha to be identifiable; mode 11 correctly records view2 as
  excluded. The follow-up accepts explicit alpha-unidentifiable entries while
  retaining their ordered finite diagnostics because coordinate inversion
  directly fits full reference flow from projected global `phi` and does not
  consume alpha values.
