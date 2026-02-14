import argparse
import os
from typing import Optional

try:
    import pandas as pd  # type: ignore[import]
except Exception:  # pragma: no cover - fallback if pandas not available
    pd = None  # type: ignore[assignment]


TEXT_EXTS = (".json", ".jsonl", ".txt")
PARQUET_EXTS = (".parquet",)


def aggregate_outputs_folder(folder: str, output_path: str) -> None:
    """Walk a folder and dump supported files into a single text file.

    Supported types:
    - .json, .jsonl, .txt are copied as-is
    - .parquet is loaded (via pandas) and pretty-printed as a table
    """
    with open(output_path, "w", encoding="utf-8") as outfile:
        for root, dirs, files in os.walk(folder):
            dirs.sort()
            files.sort()
            for name in files:
                lower = name.lower()
                if not (lower.endswith(TEXT_EXTS) or lower.endswith(PARQUET_EXTS)):
                    continue

                file_path = os.path.join(root, name)

                outfile.write("\n\n" + "=" * 50 + "\n")
                outfile.write(f"FILE: {file_path}\n")
                outfile.write("=" * 50 + "\n\n")

                if lower.endswith(PARQUET_EXTS):
                    if pd is None:
                        outfile.write(
                            "Error: pandas is not available; cannot read parquet files in this run.\n"
                        )
                        continue
                    try:
                        df = pd.read_parquet(file_path)  # type: ignore[call-arg]
                        outfile.write(df.to_string())
                        outfile.write("\n")
                    except Exception as e:  # pragma: no cover - defensive
                        outfile.write(f"Error reading parquet file: {e}\n")
                else:
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="replace") as infile:
                            outfile.write(infile.read())
                    except Exception as e:  # pragma: no cover - defensive
                        outfile.write(f"Error reading text file: {e}\n")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate all .json, .jsonl, .txt, and .parquet files under an "
            "outputs subfolder into a single text file."
        )
    )
    parser.add_argument(
        "folder",
        help=(
            "Path to the outputs subfolder to aggregate (e.g. "
            "outputs/cam03-20260103-124000_0-30s)."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        help=(
            "Optional output file path. Defaults to '<folder>/aggregated_outputs.txt'."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    folder = args.folder
    if not os.path.isdir(folder):
        raise SystemExit(f"Folder does not exist or is not a directory: {folder}")

    output_path = args.output or os.path.join(folder, "aggregated_outputs.txt")
    aggregate_outputs_folder(folder, output_path)
    print(f"Done! Wrote {output_path}.")


if __name__ == "__main__":  # pragma: no cover
    main()
