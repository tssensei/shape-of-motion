from __future__ import annotations

import argparse


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="2D modal peak-picking tools.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_mask = sub.add_parser("mask", help="Draw an ROI mask from one video frame.")
    p_mask.add_argument("--video", required=True, help="Input video path.")
    p_mask.add_argument("--t", type=float, default=0.0, help="Frame time in seconds for drawing the mask.")
    p_mask.add_argument("--resize", type=int, default=None, help="Resize max(H,W) for mask drawing.")
    p_mask.add_argument("--out", required=True, help="Output .npy mask path.")

    p_analyze = sub.add_parser("analyze", help="Analyze a video once and write an immutable modal cache.")
    p_analyze.add_argument("--video", required=True, help="Input video path.")
    p_analyze.add_argument("--cache-dir", required=True, help="New immutable modal-analysis cache directory.")
    p_analyze.add_argument("--mask", default=None, help="Optional binary ROI mask path (.npy or image).")
    p_analyze.add_argument("--t0", type=float, default=0.0, help="Clip start time in seconds.")
    p_analyze.add_argument("--t1", type=float, default=None, help="Clip end time in seconds.")
    p_analyze.add_argument("--resize", type=int, default=None, help="Resize max(H,W) before analysis.")
    p_analyze.add_argument("--max-frames", type=int, default=None, help="Optional maximum decoded frames.")
    p_analyze.add_argument("--flow-method", choices=["farneback", "tvl1"], default="farneback")
    p_analyze.add_argument("--no-smooth", action="store_true", help="Disable contrast-weighted flow smoothing.")
    p_analyze.add_argument("--sigma-b", type=float, default=3.0, help="Spatial smoothing sigma.")
    p_analyze.add_argument("--sigma-c", type=float, default=0.0, help="Reference pre-blur sigma.")
    p_analyze.add_argument("--analysis-mask-dilate-iters", type=int, default=0, help="Dilate the analysis mask with a 3x3 kernel before flow smoothing and spectrum computation.")

    p_pick = sub.add_parser("pick", help="Run interactive spectrum peak picking.")
    p_pick.add_argument("--cache-dir", required=True, help="Modal-analysis cache directory.")
    p_pick.add_argument("--snap-window-hz", type=float, default=1.0, help="Peak snapping window.")
    p_pick.add_argument("--out-json", default="outputs_modal/selected_peaks.json", help="Selected peaks JSON path.")

    p_export = sub.add_parser("export", help="Export selected complex 2D mode slices.")
    p_export.add_argument("--cache-dir", required=True, help="Modal-analysis cache directory.")
    p_export.add_argument("--out", default="outputs_modal/modal_analysis.npz", help="Output .npz path.")
    p_export.add_argument("--freqs", default=None, help="Comma-separated selected frequencies in Hz.")
    p_export.add_argument("--peaks-json", default=None, help="JSON file with selected_peaks_hz.")
    p_export.add_argument("--mode-amp-clamp", choices=["none", "local-ratio"], default="none", help="Optional robust amplitude clamp for exported complex modes.")
    p_export.add_argument("--mode-amp-local-window", type=int, default=31, help="Odd local median window for --mode-amp-clamp local-ratio.")
    p_export.add_argument("--mode-amp-ratio", type=float, default=5.0, help="Local median amplitude multiplier for --mode-amp-clamp local-ratio.")
    p_export.add_argument("--mode-amp-global-percentile", type=float, default=99.7, help="Global mask percentile cap for --mode-amp-clamp local-ratio.")

    p_inspect = sub.add_parser("inspect", help="Save non-GUI spectrum and candidate mode previews.")
    p_inspect.add_argument("--cache-dir", required=True, help="Modal-analysis cache directory.")
    p_inspect.add_argument("--out-dir", required=True, help="Output directory for PNG/JSON inspection files.")
    p_inspect.add_argument("--num-peaks", type=int, default=8, help="Number of candidate peaks to export.")
    p_inspect.add_argument("--min-freq-hz", type=float, default=0.2, help="Ignore candidates below this frequency.")
    p_inspect.add_argument("--max-freq-hz", type=float, default=None, help="Ignore candidates above this frequency.")
    p_inspect.add_argument("--peak-window-hz", type=float, default=0.6, help="Minimum spacing between candidate peaks.")
    p_inspect.add_argument("--preview-percentile", type=float, default=99.0, help="Display percentile for mode previews.")

    p_synthesize = sub.add_parser("synthesize", help="Synthesize fixed-frequency modal videos from exported .npz.")
    p_synthesize.add_argument("--modal-npz", required=True, help="modal_analysis.npz exported by the export command.")
    p_synthesize.add_argument("--out-dir", required=True, help="Output directory for synthesized videos and previews.")
    p_synthesize.add_argument("--mode-index", type=int, default=None, help="One-based mode index to synthesize. Omit to synthesize all modes.")
    p_synthesize.add_argument("--duration-s", type=float, default=4.0, help="Output video duration in seconds.")
    p_synthesize.add_argument("--fps-out", type=float, default=None, help="Output FPS. Default uses the modal-analysis FPS.")
    p_synthesize.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier for the modal oscillation.")
    p_synthesize.add_argument("--scale", type=float, default=1.0, help="Manual multiplier applied directly to raw complex mode_u/mode_v.")
    p_synthesize.add_argument("--render-backend", choices=["gl_mesh", "forward_splat", "backward_warp"], default="gl_mesh", help="Frame reconstruction backend.")
    p_synthesize.add_argument("--mesh-step", type=int, default=16, help="Regular mesh spacing in pixels for gl_mesh.")
    p_synthesize.add_argument("--show-mesh", action="store_true", help="Overlay the deformed regular mesh on gl_mesh frames.")
    p_synthesize.add_argument("--depth-weight", choices=["none", "amplitude"], default="amplitude", help="Vertex depth proxy used by gl_mesh.")
    p_synthesize.add_argument("--synth-mask-mode", choices=["none", "modal", "dilated"], default="dilated", help="Mask used to gate synthesized displacement.")
    p_synthesize.add_argument("--mask-dilate-iters", type=int, default=8, help="3x3 dilation iterations for --synth-mask-mode dilated.")
    p_synthesize.add_argument("--fill-mode", choices=["reference", "inpaint"], default="inpaint", help="Hole filling mode for forward_splat.")
    p_synthesize.add_argument("--reference-video", default=None, help="Optional video path used to load a color reference frame.")
    p_synthesize.add_argument("--reference-time-s", type=float, default=None, help="Optional reference time for --reference-video.")
    p_synthesize.add_argument("--preview-percentile", type=float, default=99.0, help="Magnitude percentile used for HSV preview images.")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.command == "mask":
        from modal_peak_pick.apps import make_mask

        make_mask.run(args)
        return
    if args.command == "analyze":
        from modal_peak_pick.apps import analyze_cache

        analyze_cache.run(args)
        return
    if args.command == "pick":
        from modal_peak_pick.apps import pick_ui

        pick_ui.run(args)
        return
    if args.command == "export":
        from modal_peak_pick.apps import export_modes

        export_modes.run(args)
        return
    if args.command == "inspect":
        from modal_peak_pick.apps import inspect_modes

        inspect_modes.run(args)
        return
    if args.command == "synthesize":
        from modal_peak_pick.apps import synthesize_mode_video

        synthesize_mode_video.run(args)
        return
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
