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

    p_pick = sub.add_parser("pick", help="Run interactive spectrum peak picking.")
    p_pick.add_argument("--video", required=True, help="Input video path.")
    p_pick.add_argument("--mask", default=None, help="Optional binary ROI mask path (.npy or image).")
    p_pick.add_argument("--t0", type=float, default=0.0, help="Clip start time in seconds.")
    p_pick.add_argument("--t1", type=float, default=None, help="Clip end time in seconds.")
    p_pick.add_argument("--resize", type=int, default=None, help="Resize max(H,W) before analysis.")
    p_pick.add_argument("--max-frames", type=int, default=None, help="Optional maximum decoded frames.")
    p_pick.add_argument("--flow-method", choices=["farneback", "tvl1"], default="farneback")
    p_pick.add_argument("--no-smooth", action="store_true", help="Disable contrast-weighted flow smoothing.")
    p_pick.add_argument("--sigma-b", type=float, default=3.0, help="Spatial smoothing sigma.")
    p_pick.add_argument("--sigma-c", type=float, default=0.0, help="Reference pre-blur sigma.")
    p_pick.add_argument("--snap-window-hz", type=float, default=1.0, help="Peak snapping window.")
    p_pick.add_argument("--out-json", default="outputs/selected_peaks.json", help="Selected peaks JSON path.")

    p_export = sub.add_parser("export", help="Export selected complex 2D mode slices.")
    p_export.add_argument("--video", required=True, help="Input video path.")
    p_export.add_argument("--out", default="outputs/modal_analysis.npz", help="Output .npz path.")
    p_export.add_argument("--freqs", default=None, help="Comma-separated selected frequencies in Hz.")
    p_export.add_argument("--peaks-json", default=None, help="JSON file with selected_peaks_hz.")
    p_export.add_argument("--peak-window-hz", type=float, default=0.6, help="Snap selected frequencies to local peaks.")
    p_export.add_argument("--mask", default=None, help="Optional binary ROI mask path (.npy or image).")
    p_export.add_argument("--t0", type=float, default=0.0, help="Clip start time in seconds.")
    p_export.add_argument("--t1", type=float, default=None, help="Clip end time in seconds.")
    p_export.add_argument("--resize", type=int, default=None, help="Resize max(H,W) before analysis.")
    p_export.add_argument("--max-frames", type=int, default=None, help="Optional maximum decoded frames.")
    p_export.add_argument("--flow-method", choices=["farneback", "tvl1"], default="farneback")
    p_export.add_argument("--no-smooth", action="store_true", help="Disable contrast-weighted flow smoothing.")
    p_export.add_argument("--sigma-b", type=float, default=3.0, help="Spatial smoothing sigma.")
    p_export.add_argument("--sigma-c", type=float, default=0.0, help="Reference pre-blur sigma.")

    p_inspect = sub.add_parser("inspect", help="Save non-GUI spectrum and candidate mode previews.")
    p_inspect.add_argument("--video", required=True, help="Input video path.")
    p_inspect.add_argument("--out-dir", required=True, help="Output directory for PNG/JSON inspection files.")
    p_inspect.add_argument("--mask", default=None, help="Optional binary ROI mask path (.npy or image).")
    p_inspect.add_argument("--t0", type=float, default=0.0, help="Clip start time in seconds.")
    p_inspect.add_argument("--t1", type=float, default=None, help="Clip end time in seconds.")
    p_inspect.add_argument("--resize", type=int, default=None, help="Resize max(H,W) before analysis.")
    p_inspect.add_argument("--max-frames", type=int, default=None, help="Optional maximum decoded frames.")
    p_inspect.add_argument("--flow-method", choices=["farneback", "tvl1"], default="farneback")
    p_inspect.add_argument("--no-smooth", action="store_true", help="Disable contrast-weighted flow smoothing.")
    p_inspect.add_argument("--sigma-b", type=float, default=3.0, help="Spatial smoothing sigma.")
    p_inspect.add_argument("--sigma-c", type=float, default=0.0, help="Reference pre-blur sigma.")
    p_inspect.add_argument("--num-peaks", type=int, default=8, help="Number of candidate peaks to export.")
    p_inspect.add_argument("--min-freq-hz", type=float, default=0.2, help="Ignore candidates below this frequency.")
    p_inspect.add_argument("--max-freq-hz", type=float, default=None, help="Ignore candidates above this frequency.")
    p_inspect.add_argument("--peak-window-hz", type=float, default=0.6, help="Minimum spacing between candidate peaks.")
    p_inspect.add_argument("--preview-percentile", type=float, default=99.0, help="Display percentile for mode previews.")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.command == "mask":
        from modal_peak_pick.apps import make_mask

        make_mask.run(args)
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
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
