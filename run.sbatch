#!/bin/bash
#SBATCH --job-name=bush4-723
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/home/zs292/slurm_outputs/som/bush4-723-%j.out
#SBATCH --error=/home/zs292/slurm_outputs/som/bush4-723-%j.err

set -euo pipefail

source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate som

cd /home/zs292/shape-of-motion

SCENE_ROOT=/home/zs292/data_formal/bush4_3view_v1
BASIS_ROOT=$SCENE_ROOT/bases/greedy_0p2_4p0_step0p025_k60prefix_v1
FIELD_ROOT=$SCENE_ROOT/modal_fields/greedy_0p2_4p0_step0p025_k60prefix_v1

INPUT_CKPT=$SCENE_ROOT/shared/inputs/static_checkpoint/bush4_static_for_modal_solver.ckpt
VIEW_CONFIG_DIR=$SCENE_ROOT/shared/inputs/view_configs
MODAL_ANALYSIS_DIR=$BASIS_ROOT/modal_analysis
OBSERVATION_TOPOLOGY=$BASIS_ROOT/gaussian_observations/topology.npz
MEASUREMENT_DIR=$BASIS_ROOT/gaussian_observations/measurements
OBSERVED_GRAPH=$SCENE_ROOT/shared/preprocessing/observed_structure_graphs/rgb_depth_knn8_maxdist0p008_v3/graph.npz
MOTION_FILL_GRAPH=$SCENE_ROOT/shared/preprocessing/motion_fill_graphs/knn8_maxdist0p008_v1/graph.npz

OUTPUT_NAME=rigid_rgbdepth_v3_sequential_k60_bounded_complex_v1
OUTPUT_DIR=$FIELD_ROOT/$OUTPUT_NAME
LOG_DIR=$FIELD_ROOT/logs
LOG_PATH=$LOG_DIR/$OUTPUT_NAME.log

mkdir -p "$LOG_DIR"

REQUIRED_FILES=(
  "$INPUT_CKPT"
  "$VIEW_CONFIG_DIR/view1_config.json"
  "$VIEW_CONFIG_DIR/view2_config.json"
  "$VIEW_CONFIG_DIR/view3_config.json"
  "$MODAL_ANALYSIS_DIR/view1_modal_analysis.npz"
  "$MODAL_ANALYSIS_DIR/view2_modal_analysis.npz"
  "$MODAL_ANALYSIS_DIR/view3_modal_analysis.npz"
  "$OBSERVATION_TOPOLOGY"
  "$OBSERVED_GRAPH"
  "$MOTION_FILL_GRAPH"
)
for path in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required input: $path" >&2
    exit 1
  fi
done

shopt -s nullglob
MEASUREMENT_ARGS=()
for mode_index in $(seq 0 59); do
  mode_tag=$(printf '%03d' "$mode_index")
  matches=("$MEASUREMENT_DIR"/mode_"$mode_tag"_*.npz)
  if [[ "${#matches[@]}" -ne 1 ]]; then
    echo "Expected exactly one measurement for mode $mode_index, found ${#matches[@]}" >&2
    exit 1
  fi
  MEASUREMENT_ARGS+=(
    --rigid-component-observation-measurement "${matches[0]}"
  )
done
shopt -u nullglob

echo "Validated ${#REQUIRED_FILES[@]} shared inputs and $((${#MEASUREMENT_ARGS[@]} / 2)) measurements"

COMMON_ARGS=(
  --input-ckpt "$INPUT_CKPT"
  --view-config "$VIEW_CONFIG_DIR/view1_config.json"
  --view-config "$VIEW_CONFIG_DIR/view2_config.json"
  --view-config "$VIEW_CONFIG_DIR/view3_config.json"
  --modal-npz "$MODAL_ANALYSIS_DIR/view1_modal_analysis.npz"
  --modal-npz "$MODAL_ANALYSIS_DIR/view2_modal_analysis.npz"
  --modal-npz "$MODAL_ANALYSIS_DIR/view3_modal_analysis.npz"
  --mode-indices all
  --mask-erode-iters 1
  --pixel-sample-stride 2
  --pixel-candidate-k 4
  --pixel-preselect-k 32
  --pixel-render-acc-min 0.05
  --pixel-min-contribution 1e-12
  --freq-tolerance-hz 1e-6
  --solve-method rigid-components
  --rigid-component-graph "$OBSERVED_GRAPH"
  --rigid-component-observation-topology "$OBSERVATION_TOPOLOGY"
  --alpha-gain-min 0.25
  --alpha-gain-max 4.0
  --alpha-min-shared-points 16
  --alpha-rank-ratio-min 1e-4
  --alpha-info-ratio-min 1e-4
  --alpha-failure exclude
  --rigid-component-rcond 1e-8
  --rigid-seed-min-valid-views 2
  --rigid-seed-min-secondary-view-node-ratio 0.3333333333333333
  --rigid-seed-min-singular-ratio 1e-3
  --rigid-seed-max-finite-drift 2.0
  --rigid-motion-fill-stage sequential
  --rigid-single-view-observable-ratio 1e-2
  --rigid-single-view-ray-direction-min-fraction 0.8
  --motion-fill
  --motion-fill-k 8
  --motion-fill-max-distance 0.008
  --motion-fill-graph "$MOTION_FILL_GRAPH"
  --motion-fill-max-anchor-hops 8
)

if [[ -f "$OUTPUT_DIR/modal_modes_manifest.json" ]]; then
  echo "[$(date --iso-8601=seconds)] Skipping completed bounded-complex -> $OUTPUT_DIR"
else
  echo "[$(date --iso-8601=seconds)] Starting bounded-complex -> $OUTPUT_DIR"
  PYTHONPATH=. python -u -m modal_surface \
    "${COMMON_ARGS[@]}" \
    "${MEASUREMENT_ARGS[@]}" \
    --out-dir "$OUTPUT_DIR" \
    --resume \
    --alpha-model bounded-complex \
    2>&1 | tee "$LOG_PATH"
  echo "[$(date --iso-8601=seconds)] Finished bounded-complex -> $OUTPUT_DIR"
fi
