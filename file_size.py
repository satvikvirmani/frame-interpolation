import os
from PIL import Image
from tqdm import tqdm

def get_unique_image_sizes(root_dir, list_file):
    """
    Scans all image folders in Vimeo-90K train list and finds distinct resolutions.
    """
    list_path = os.path.join(root_dir, list_file)
    with open(list_path, 'r') as f:
        samples = [line.strip() for line in f.readlines() if line.strip()]

    unique_sizes = set()

    for seq in tqdm(samples):
        seq_path = os.path.join(root_dir, 'sequences', seq)
        for img_name in ['im1.png', 'im2.png', 'im3.png']:
            img_path = os.path.join(seq_path, img_name)
            if os.path.exists(img_path):
                with Image.open(img_path) as img:
                    unique_sizes.add(img.size)

    return unique_sizes


if __name__ == "__main__":
    root_dir = "vimeo_triplet"
    list_file = "tri_trainlist.txt"

    print("Scanning image sizes in training list...")
    unique_sizes = get_unique_image_sizes(root_dir, list_file)

    print("\nDistinct image dimensions found:")
    for w, h in sorted(unique_sizes):
        print(f" - {w} × {h}")

    print(f"\nTotal distinct sizes: {len(unique_sizes)}")