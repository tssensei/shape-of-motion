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
