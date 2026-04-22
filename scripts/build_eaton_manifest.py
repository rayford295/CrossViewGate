from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crossview_conflict.data.manifests import build_altadena_manifest, build_eaton_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CrossViewConflict manifests from local disaster datasets.")
    parser.add_argument(
        "--source",
        choices=["altadena", "eaton"],
        required=True,
        help="Which dataset schema to parse.",
    )
    parser.add_argument("--index-csv", required=True, help="Input CSV index path.")
    parser.add_argument(
        "--dataset-root",
        help="Dataset root used to resolve relative image paths. Required for Altadena manifests.",
    )
    parser.add_argument("--output-csv", required=True, help="Output manifest CSV path.")
    parser.add_argument(
        "--binary-scheme",
        default="operational",
        choices=["operational", "sensitive"],
        help="Binary label mapping for triage.",
    )
    parser.add_argument(
        "--allow-missing-pairs",
        action="store_true",
        help="Keep Altadena rows even if one side of the image pair is missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source == "altadena":
        if not args.dataset_root:
            raise ValueError("--dataset-root is required for Altadena manifests.")
        manifest = build_altadena_manifest(
            index_csv=args.index_csv,
            dataset_root=args.dataset_root,
            output_csv=args.output_csv,
            binary_scheme=args.binary_scheme,
            require_complete_pairs=not args.allow_missing_pairs,
        )
    else:
        manifest = build_eaton_manifest(
            attachments_index_csv=args.index_csv,
            output_csv=args.output_csv,
            binary_scheme=args.binary_scheme,
        )
    print(f"Wrote {len(manifest)} rows to {Path(args.output_csv).resolve()}")


if __name__ == "__main__":
    main()
