import os

# Configuration
SOURCE_DIR = '.'  # Current directory
OUTPUT_FILE = 'project_code_dump.txt'
# Add folders to ignore
IGNORE_DIRS = {'.git', '__pycache__', 'venv', 'env', '.idea', '.vscode', 'node_modules', 'build', 'dist', 'Instances'}
# Add extensions you want to include
INCLUDE_EXTS = {'.py', '.html', '.css', '.js', '.json', '.md', '.sql'}

def collect_code():
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        for root, dirs, files in os.walk(SOURCE_DIR):
            # Modify dirs in-place to skip ignored directories
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            
            for file in files:
                ext = os.path.splitext(file)[1]
                if ext in INCLUDE_EXTS:
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            outfile.write(f"\n{'='*20}\nFILE: {file_path}\n{'='*20}\n")
                            outfile.write(infile.read())
                            outfile.write("\n")
                    except Exception as e:
                        print(f"Skipping {file}: {e}")
    print(f"Done! Upload {OUTPUT_FILE} to the chat.")

if __name__ == '__main__':
    collect_code()