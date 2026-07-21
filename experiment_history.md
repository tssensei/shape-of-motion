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
