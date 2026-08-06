# Poster Runbook

## Status

This package is prepared for a **30 in W x 40 in H portrait poster**. The Paper2Poster command below has **not** been executed. The source brief still contains figure placeholders, so generating the final PPTX before exporting the Bush assets is expected to produce an incomplete visual result.

## Source material

- Main source: `docs/poster_brief/shape_of_motion_poster_brief.tex`
- References: `docs/poster_brief/references.bib`
- Figure requirements: `docs/poster_brief/figures/README.md`
- Condensed content: `docs/poster_package/bush_internal_v1/poster_copy.md`
- Layout instructions: `docs/poster_package/bush_internal_v1/poster_layout.md`
- Style override: `docs/poster_package/bush_internal_v1/poster.yaml`

## Step 1 -- Export and verify Bush assets

Create these files under `docs/poster_brief/figures/`:

```text
bush_capture_and_cameras.png
bush_static_3dgs.png
bush_flow_and_frequency.png
bush_rigid_components.png
bush_motion_fill.png
bush_modes_grid.png
```

Before accepting an asset:

- verify that the screenshot comes from the intended Bush checkpoint/manifest;
- record the source path and Viewer command separately for reproducibility;
- crop UI chrome and empty space;
- check that text remains legible at poster scale; and
- use consistent camera and modal visualization settings across the mode grid.

## Step 2 -- Prepare the Paper2Poster input PDF

Compile the source only after the figures have been inserted and inspected. From `docs/poster_brief/`, a typical TeX installation can use:

```bash
latexmk -pdf shape_of_motion_poster_brief.tex
```

The expected result is:

```text
docs/poster_brief/shape_of_motion_poster_brief.pdf
```

Paper2Poster expects the input file to be named `paper.pdf` inside a paper-specific directory:

```text
Paper2Poster-data/
└── bush_internal_v1/
    ├── paper.pdf
    └── poster.yaml
```

Copy the compiled PDF to `Paper2Poster-data/bush_internal_v1/paper.pdf` and copy this package's `poster.yaml` beside it.

## Step 3 -- Prepare Paper2Poster

In a separate checkout of the official Paper2Poster repository:

```bash
pip install -r requirements.txt
```

Paper2Poster also requires LibreOffice and Poppler. When using the API-based `4o` model configuration, create a `.env` file in the Paper2Poster repository root containing a valid API key:

```text
OPENAI_API_KEY=<your_key>
```

Do not commit `.env` or paste the key into the run command.

## Step 4 -- Generate the editable PPTX

Run from the Paper2Poster repository root after replacing `DATASET_DIR` with an absolute path:

```bash
python -m PosterAgent.new_pipeline \
  --poster_path="DATASET_DIR/bush_internal_v1/paper.pdf" \
  --model_name_t="4o" \
  --model_name_v="4o" \
  --poster_width_inches=30 \
  --poster_height_inches=40 \
  --max_workers=4
```

This intentionally omits conference venue and logo options. The generated `poster.pptx` should remain editable. Paper2Poster controls the output-directory naming from the selected model configuration; inspect the terminal's final saved path rather than assuming a fixed absolute location.

## Step 5 -- Manual PPTX pass

Paper2Poster output is a starting point. In PowerPoint:

1. set or verify the page size is exactly 30 x 40 inches in portrait orientation;
2. verify there are no invented authors, affiliations, metrics, or claims;
3. replace any AI-redrawn scientific diagram whose geometry or notation changed;
4. compare all equations against `shape_of_motion_poster_brief.tex`;
5. ensure the Bush mode images use the same camera, scale, and color convention;
6. shorten text rather than shrinking body copy below approximately 30 pt;
7. keep the hero pipeline and Bush mode grid visually dominant;
8. confirm that the narrative stops at the completed spatial 3D modes; and
9. export a PDF and inspect it at 100% zoom before printing.

## Final grounding checklist

- [ ] Every displayed equation agrees with the source brief.
- [ ] Every frequency label comes from the selected-frequency artifact.
- [ ] Every Bush image has a recorded checkpoint/manifest source.
- [ ] No Corn, curtain, soft-elastic, physics, coordinate-fit, or reconstruction result appears.
- [ ] No quantitative comparison is implied.
- [ ] The limitation about Fourier patterns versus physical eigenmodes is visible.
- [ ] No credentials or cluster paths appear in the poster.
