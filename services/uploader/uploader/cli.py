from __future__ import annotations

import argparse


from .config import load_config
from .service import run_upload
from .worker import run_worker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        help="Path to export_manifest.jsonl",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help="Run uploader as polling worker",
    )

    args = parser.parse_args()

    cfg = load_config()

    if args.worker:
        run_worker(cfg)
    else:
        if not args.manifest:
            raise SystemExit("--manifest required unless --worker is used")
        run_upload(args.manifest, cfg)


if __name__ == "__main__":
    main()
