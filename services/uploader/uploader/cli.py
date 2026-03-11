from __future__ import annotations

import argparse

from .config import load_config
from .service import run_upload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload Stage F exports to Supabase storage and tables."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to Stage F export_manifest.jsonl",
    )
    args = parser.parse_args()

    cfg = load_config()
    run_upload(args.manifest, cfg)


if __name__ == "__main__":
    main()
