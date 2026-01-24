import os
import re
from pathlib import Path

DOCS_ROOT = Path("docs")

# Regex for markdown links: [text](target)
LINK_RE = re.compile(r'(\[([^\]]+)\]\(([^)]+)\))')

def is_internal_link(target):
    if target.startswith("http") or target.startswith("//") or target.startswith("#"):
        return False
    if target.startswith("mailto:"):
        return False
    return True

def normalize_target(md_path, target):
    # Remove any anchor for path normalization, add back later
    anchor = ''
    if '#' in target:
        target, anchor = target.split('#', 1)
        anchor = '#' + anchor

    # Remove leading ./ or ../ for normalization
    abs_target = (md_path.parent / target).resolve() if not target.startswith('/') else DOCS_ROOT / target.lstrip('/')
    abs_target = abs_target.relative_to(DOCS_ROOT.resolve())

    # If it's a markdown file, convert to .html
    if abs_target.suffix == '.md':
        abs_target = abs_target.with_suffix('.html')

    # If it's a directory or index, ensure trailing slash
    if abs_target.suffix == '' and not str(abs_target).endswith('/'):
        abs_target = Path(str(abs_target) + '/')

    # Always start with /
    norm_path = '/' + str(abs_target)
    return f"{{{{ '{norm_path}{anchor}' | relative_url }}}}"

def process_file(md_path):
    with open(md_path, encoding='utf-8') as f:
        lines = f.readlines()

    in_code_block = False
    changed = False
    replacements = []

    def replace_link(match):
        full, text, target = match.group(0), match.group(2), match.group(3)
        if not is_internal_link(target):
            return full
        try:
            new_link = normalize_target(md_path, target)
            replacements.append((target, new_link))
            return f"[{text}]({new_link})"
        except Exception as e:
            print(f"WARNING: Could not normalize link '{target}' in {md_path}: {e}")
            return full

    new_lines = []
    for line in lines:
        if line.strip().startswith("```") or line.strip().startswith("~~~"):
            in_code_block = not in_code_block
            new_lines.append(line)
            continue
        if in_code_block:
            new_lines.append(line)
            continue
        new_line = LINK_RE.sub(replace_link, line)
        if new_line != line:
            changed = True
        new_lines.append(new_line)

    if changed:
        with open(md_path, "w", encoding='utf-8') as f:
            f.writelines(new_lines)
    return changed, replacements

def main():
    changed_files = []
    all_replacements = {}
    for root, dirs, files in os.walk(DOCS_ROOT):
        for file in files:
            if file.endswith(".md"):
                md_path = Path(root) / file
                changed, replacements = process_file(md_path)
                if changed:
                    changed_files.append(str(md_path))
                    all_replacements[str(md_path)] = replacements

    print("Changed files:")
    for f in changed_files:
        print(f"  {f}")
        for orig, new in all_replacements[f]:
            print(f"    {orig} -> {new}")

if __name__ == "__main__":
    main()
