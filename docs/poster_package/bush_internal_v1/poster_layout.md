# Poster Layout

## Canvas

- Physical size: **30 in W x 40 in H**
- Orientation: portrait
- Outer margin: 0.55 in
- Header height: approximately 3.8 in
- Footer/takeaway band: approximately 2.4 in
- Main body: three columns with 0.35 in gutters
- Approximate column width: 9.4 in

The poster should read top-to-bottom within each local block while maintaining a clear overall path from left to right:

**capture and representation -> image evidence and correspondence -> 3D solve and Bush modes**

## Header

- Full-width title on no more than two lines.
- One-sentence objective directly below the title.
- No author, affiliation, logo, or venue strip in this internal version.
- Use a thin four-color pipeline accent: camera blue, geometry teal, evidence orange, mode purple.

## Upper hero region

Place a simplified pipeline diagram immediately below the header, spanning all three columns and approximately 4.5--5.0 in high:

1. sweep + fixed-view videos;
2. joint COLMAP;
3. canonical foreground/background 3DGS;
4. masked optical flow and selected frequencies;
5. pixel--Gaussian observation topology;
6. local component projection solve; and
7. motion fill to the complete foreground.

This is the first visual viewers should notice. Use arrows and color-coded stages rather than prose inside the diagram.

## Left column: capture and 3D representation

### Block A -- Why is this hard?

- Four short bullets from `poster_copy.md`.
- Add one small schematic showing that a camera sees only a 2D projection of a 3D displacement.

### Block B -- Shared cameras

- Use `bush_capture_and_cameras.png` at nearly full column width.
- Caption: sweep and fixed references are registered in one COLMAP coordinate system.

### Block C -- 3D Gaussian Splatting

- Use `bush_static_3dgs.png` beside a compact Gaussian-primitive schematic.
- Retain only the primitive and compositing equations.
- Put the plain-language ellipsoid explanation in a tinted callout.

## Center column: vibration evidence and correspondence

### Block D -- From video to frequency

- Use `bush_flow_and_frequency.png` at full column width.
- Show the Fourier equation directly below it.
- Label the complex field as 2D evidence, not a 3D mode.

### Block E -- Pixel--Gaussian correspondence

- This is the main technical block.
- Use a compact diagram: pixel -> rendered-depth unprojection -> foreground KNN -> Gaussian score -> camera reprojection.
- Retain the Gaussian score equation.
- Include a visually separated sentence explaining why appearance compositing is not a unique motion correspondence.

## Right column: structure, solve, and result

### Block F -- Local structure and 3D solve

- Place `bush_rigid_components.png` above the solve equation.
- Use a three-stage inset: mutual KNN -> remove color/depth discontinuities -> connected local components.
- Retain the weighted projected component objective as the main mathematical equation.
- State explicitly that the whole Bush is not forced to be one rigid body.

### Block G -- Motion fill

- Use `bush_motion_fill.png` as a before/after pair with identical view and color scale.
- Limit explanation to two bullets: trusted seeds remain fixed; graph distance limits propagation.

### Block H -- Bush spatial modes

- Use `bush_modes_grid.png` as the largest result image, ideally 8.8--9.2 in wide.
- Four same-camera views with explicit frequency labels and one shared phase/amplitude legend.
- Do not show terminal or Viewer chrome.
- Caption only what is visible; do not claim quantitative superiority.

## Bottom takeaway band

Use three numbered statements:

1. Build one shared 3D scene.
2. Lift multi-view image vibrations into 3D.
3. Use local structure and graph completion to obtain one displacement per foreground Gaussian.

Place the limitation about non-normalized eigenmodes in smaller text below the takeaway. Omit a QR/contact placeholder until a real destination is supplied.

## Visual hierarchy

- Title: 72--84 pt
- Section titles: 46--54 pt
- Body: 32--38 pt
- Captions: 25--29 pt
- Equations: large enough to read from approximately 4 ft; prefer one equation per block
- Use white background, dark text, and one accent color per pipeline stage
- Avoid shaded boxes around every section; reserve tinted backgrounds for the plain-language and limitation callouts

## Asset QA

- Export raster figures at least 1800 px on the long side.
- Keep one camera, Gaussian scale, opacity, amplitude normalization, and phase convention across modal screenshots.
- Hide background Gaussians for component/mode diagnostics when they obscure the foreground; retain them for the canonical RGB scene.
- Crop large empty Viser regions and all terminal/browser chrome.
- Never mix RGB, component, and modal-phase colors without a label.
