import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd


def parquet_to_jsonl(src: Path, dst: Path) -> None:
    df = pd.read_parquet(src)
    with dst.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            obj = row.to_dict()
            f.write(json.dumps(obj, default=str) + "\n")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Convert a Parquet file to JSONL.")
    parser.add_argument("src", help="Source Parquet file")
    parser.add_argument("dst", nargs="?", help="Destination JSONL file (optional)")
    args = parser.parse_args(argv)

    src_path = Path(args.src)
    if not src_path.is_file():
        raise SystemExit(f"Source not found: {src_path}")

    dst_path = Path(args.dst) if args.dst else src_path.with_suffix(".jsonl")
    parquet_to_jsonl(src_path, dst_path)
    print(f"Wrote {dst_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
