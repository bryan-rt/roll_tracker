import os

folders_to_audit = ['src', 'configs', 'services', 'tools', 'bin', 'tests', 'docs', 'planning']
output_file = 'full_codebase.txt'
exts = ('.py', '.yaml', '.yml', '.json', '.txt', '.sh', '.dockerfile', '.md')

def main() -> None:
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for folder in folders_to_audit:
            if not os.path.isdir(folder):
                continue
            for root, dirs, files in os.walk(folder):
                dirs.sort()
                files.sort()
                for file in files:
                    if not file.lower().endswith(exts):
                        continue
                    file_path = os.path.join(root, file)
                    outfile.write("\n\n" + '='*50 + "\n")
                    outfile.write(f"FILE: {file_path}\n")
                    outfile.write('='*50 + "\n\n")
                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            outfile.write(infile.read())
                    except Exception as e:
                        outfile.write(f"Error reading file: {e}\n")

if __name__ == '__main__':
    main()
    print(f"Done! Wrote {output_file}.")
