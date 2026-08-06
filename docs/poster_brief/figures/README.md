# Bush figure asset checklist

The LaTeX brief compiles without these images by showing labeled placeholders.
Replace each placeholder by exporting the requested Bush result to the exact
filename below. Prefer PNG for plots and UI captures; use PDF for vector
diagrams if a later replacement is drawn manually.

## Version-2 poster placement

`docs/poster_drafts/bush_spatial_modes_poster_v2.tex` uses the same stable
filenames but gives more space to background and assumptions. The most
important final assets are, in order:

1. `bush_flow_and_frequency.png`, which supports the Davis image-space modal
   analysis explanation;
2. `bush_capture_and_cameras.png`, which shows the asynchronous capture and
   shared COLMAP coordinate system;
3. `bush_rigid_components.png`, which illustrates the local 3D solve; and
4. `bush_modes_grid.png`, which is the largest qualitative result panel.

`bush_static_3dgs.png` remains a smaller background panel. The motion-fill
before/after image is optional in this layout because the full pipeline and
mode grid have higher priority. Use muted annotations that match the poster's
slate, sage, clay, and warm-gray palette; do not add bright rainbow borders.

Keep screenshots tightly cropped. Avoid terminal chrome and large empty Viewer
regions. For all modal screenshots, use the same camera, Viewer resolution,
Gaussian scale, opacity, amplitude normalization, and phase-color convention.

| Filename | Content | Suggested composition |
| --- | --- | --- |
| `bush_capture_and_cameras.png` | Capture inputs and shared camera geometry | One sweep frame, the three fixed-view reference frames, and a Viewer screenshot with registered camera frustums around the static scene. Add short labels instead of a long caption inside the image. |
| `bush_static_3dgs.png` | Canonical Bush 3DGS | A clean novel-view render. Optionally place one real sweep frame beside it for a visual input/render comparison. Use normal RGB color and hide unrelated modal controls. |
| `bush_flow_and_frequency.png` | 2D vibration evidence | A three-panel image: masked optical-flow visualization, power spectrum with one selected frequency, and the matching complex modal image. Use the same fixed camera in all panels. |
| `bush_rigid_components.png` | Observed structure graph | The graph Viewer colored by connected component, with observed nodes or foreground splats faintly visible for context. Choose a view that separates branches and shows that the plant is divided into local pieces rather than one global rigid body. |
| `bush_motion_fill.png` | Coverage before and after motion fill | The same frequency and camera before/after completion. Use identical bounds and color scale. If possible, add a small legend for observed seed, filled, and unobserved roles. |
| `bush_modes_grid.png` | Four representative completed 3D modes | Four same-camera modal-phase or displacement screenshots with frequency labels. The current `.tex` contains four independent boxes; when this grid is ready, either replace those boxes with one `\includegraphics` call or split the grid into four files. |

## Recommended poster crops

- Export at least 1800 pixels on the long side; the eventual 30-by-40-inch
  poster will expose low-resolution screenshots.
- Prefer the static camera angle that makes foreground branches readable in
  silhouette.
- Hide background Gaussians for diagnostic graph/mode images when they obscure
  the foreground. Keep background visible in the canonical RGB result so the
  scene still reads naturally.
- Do not mix `modal phase`, RGB, and component colors without an explicit label.
- Keep this version qualitative; the intended result is an accessible visual
  understanding of the recovered 3D spatial modes.

## Source locations to record during export

Before final poster assembly, add the exact Bush checkpoint, modal manifest,
graph NPZ, selected-frequency JSON, and Viewer command used for every image.
This makes the figure set reproducible without putting cluster paths into the
poster itself.
