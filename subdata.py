import os
import shutil
from tqdm import tqdm

# --- Configuration ---
SOURCE_PARENT_DIR = 'vimeo_triplet'
DEST_PARENT_DIR = 'vimeo_triplet2'
NUM_FOLDERS_TO_COPY = 800

# --- Paths ---
# Source paths
source_list_file = os.path.join(SOURCE_PARENT_DIR, 'tri_trainlist.txt')
source_sequences_dir = os.path.join(SOURCE_PARENT_DIR, 'sequences')

# Destination paths
dest_list_file = os.path.join(DEST_PARENT_DIR, 'tri_trainlist.txt')
dest_sequences_dir = os.path.join(DEST_PARENT_DIR, 'sequences')

def create_dataset_subset():
    """
    Reads the first N lines of a list file, copies the corresponding folders
    to a new directory, and creates a new list file.
    """
    print(f"🚀 Starting to create a subset dataset in '{DEST_PARENT_DIR}'...")

    # 1. Create the main destination directories
    print(f"Creating destination folder: {dest_sequences_dir}")
    os.makedirs(dest_sequences_dir, exist_ok=True)

    # 2. Read the first N lines from the source list file
    try:
        with open(source_list_file, 'r') as f:
            # Read all lines and slice the first N
            folder_paths = [line.strip() for line in f.readlines()]
            paths_to_copy = folder_paths[:NUM_FOLDERS_TO_COPY]
            print(f"Found {len(paths_to_copy)} folder paths to copy (up to a max of {NUM_FOLDERS_TO_COPY}).")
    except FileNotFoundError:
        print(f"❌ ERROR: Source list file not found at '{source_list_file}'")
        return

    # 3. Copy each folder
    print("Copying sequence folders...")
    copied_paths = []
    # Using tqdm for a nice progress bar
    for relative_path in tqdm(paths_to_copy, unit="folder"):
        source_folder = os.path.join(source_sequences_dir, relative_path)
        dest_folder = os.path.join(dest_sequences_dir, relative_path)

        # Ensure the parent directory in the destination exists (e.g., .../sequences/00001/)
        os.makedirs(os.path.dirname(dest_folder), exist_ok=True)
        
        # Check if source exists before copying
        if os.path.exists(source_folder):
            try:
                shutil.copytree(source_folder, dest_folder)
                copied_paths.append(relative_path)
            except FileExistsError:
                # This allows the script to be re-run without erroring on existing folders
                # print(f"  - Warning: Destination '{dest_folder}' already exists. Skipping.")
                copied_paths.append(relative_path) # Assume it's already correctly copied
            except Exception as e:
                print(f"  - ❌ Error copying '{source_folder}': {e}")
        else:
            print(f"  - ⚠️ Warning: Source folder '{source_folder}' not found. Skipping.")

    # 4. Write the new list file with only the paths that were successfully processed
    if not copied_paths:
        print("No folders were copied. The new list file will be empty.")
        return
        
    print(f"\nWriting new list file to '{dest_list_file}'...")
    with open(dest_list_file, 'w') as f:
        for path in copied_paths:
            f.write(path + '\n')

    print(f"\n✅ Success! Created a subset with {len(copied_paths)} folders in '{DEST_PARENT_DIR}'.")

if __name__ == '__main__':
    create_dataset_subset()