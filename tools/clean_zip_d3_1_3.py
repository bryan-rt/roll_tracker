import zipfile
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZIP_NAME = "D3_1_3.zip"

zip_path = ROOT / ZIP_NAME
print("zip_path", zip_path)
if not zip_path.exists():
    raise SystemExit(f"missing {zip_path}")

backup = ROOT / "D3_1_3_python_clean_backup.zip"
if not backup.exists():
    shutil.copy2(zip_path, backup)
    print("backup_created", backup)
else:
    print("backup_already_exists", backup)

keep_ext = {".py", ".json", ".jsonl", ".yaml", ".yml"}

tmp_zip = ROOT / "D3_1_3.tmp.zip"

with zipfile.ZipFile(zip_path, "r") as zin, zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zout:
    for info in zin.infolist():
        name = info.filename
        # Skip directories
        if name.endswith("/"):
            continue
        base = os.path.basename(name)
        # Skip macOS metadata
        if base in {".DS_Store"} or base.startswith("._") or name.startswith("__MACOSX/"):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext not in keep_ext:
            continue
        data = zin.read(name)
        # Preserve relative path but drop original ZipInfo metadata
        zout.writestr(name, data)

shutil.move(tmp_zip, zip_path)
print("cleaned", zip_path)
