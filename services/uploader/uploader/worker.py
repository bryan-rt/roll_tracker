import time
from pathlib import Path

from .service import run_upload


def discover_manifests(scan_root: str):
    root = Path(scan_root)
    return list(root.glob("**/stage_F/export_manifest.jsonl"))


def run_worker(cfg):

    print("[uploader] worker started")
    print(f"[uploader] scanning root: {cfg.scan_root}")

    while True:

        manifests = discover_manifests(cfg.scan_root)

        if manifests:
            print(f"[uploader] discovered {len(manifests)} manifest(s)")

        for manifest in manifests:

            try:

                print(f"[uploader] processing manifest {manifest}")

                run_upload(str(manifest), cfg)

                manifest.unlink(missing_ok=True)

                print(f"[uploader] manifest complete, removed {manifest}")

            except Exception as e:

                print(f"[uploader] error processing {manifest}: {e}")

        time.sleep(cfg.poll_seconds)

