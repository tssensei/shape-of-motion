# Poster Brief

## Working title

**Recovering 3D Vibration Modes from Multi-View Videos with Gaussian Splatting**

Optional short subtitle: **From image-space optical flow to spatial motion fields on a shared 3D scene**

## Format and audience

- Size: **30 inches wide x 40 inches high**
- Orientation: portrait
- Setting: informal internal university research event
- Audience: mixed technical; most viewers have general computer-science knowledge, but 3D Gaussian Splatting cannot be assumed
- Tone: technically credible, visual, and easy to scan
- Target visible body copy: approximately 400--500 words, excluding equations and captions
- Branding: no author, institution, conference-logo, sponsor, or funding block unless supplied later

## Scope

This version follows only the Bush experiment and stops after recovering the completed spatial 3D mode fields

\[
\phi_k\in\mathbb C^{G\times 3}.
\]

It deliberately excludes:

- Corn and curtain experiments;
- soft-elastic or staged-solver comparisons;
- per-frame modal-coordinate recovery;
- physics post-fitting or oscillator playback;
- quantitative evaluation; and
- reconstructed-video results.

## Main takeaway

A moving sweep camera supplies a canonical Gaussian-splat scene, while synchronized fixed views supply 2D vibration evidence. Joint camera calibration, explicit pixel--Gaussian correspondence, local color/depth-filtered structure, and multi-view reprojection lift selected image frequencies into spatially coherent 3D displacement fields attached to the foreground Gaussians.

## Narrative priorities

1. **Why it is difficult:** each camera measures only 2D motion, while depth motion is ambiguous.
2. **Why 3DGS helps:** it supplies an explicit, renderable set of 3D locations on which motion can live.
3. **How evidence is connected to geometry:** rendered depth and Gaussian scoring associate masked flow pixels with plausible foreground Gaussians.
4. **How 3D motion is solved:** nearby, appearance-consistent Gaussians form local components whose projected complex motion is fitted across views.
5. **How coverage is completed:** graph-based motion fill propagates trusted solutions to foreground Gaussians without adequate direct observations.

## Required visuals

| Priority | Asset | Purpose |
| --- | --- | --- |
| 1 | `bush_pipeline_overview` | Hero diagram linking capture, shared cameras, 3DGS, flow spectra, topology, local solve, and motion fill. |
| 2 | `bush_modes_grid.png` | Main qualitative result: four completed modes from one consistent camera and color convention. |
| 3 | `bush_capture_and_cameras.png` | Sweep frame, fixed references, and registered camera frustums in one coordinate system. |
| 4 | `bush_flow_and_frequency.png` | Masked flow, spectrum, and one selected complex modal image. |
| 5 | `bush_rigid_components.png` | Color/depth-filtered observed structure graph colored by connected component. |
| 6 | `bush_motion_fill.png` | Same mode before and after graph completion with a shared scale. |
| 7 | `bush_static_3dgs.png` | Clean RGB render of the canonical foreground/background scene. |

## Grounding and unresolved items

- No quantitative performance claim is available for this version.
- Exact selected Bush frequencies are intentionally omitted until the final four result images are chosen.
- Author/contact text and QR destination were not supplied and will not be invented.
- The six Bush assets listed in `docs/poster_brief/figures/README.md` still need to be exported and checked.
- The recovered Fourier fields are evidence-driven spatial vibration patterns; they are not claimed to be physically normalized structural eigenmodes.
