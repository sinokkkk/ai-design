from pathlib import Path
from PIL import Image

def convert_folder(folder_path):
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Folder not found: {folder}")
        return

    ppm_files = list(folder.glob("*.ppm"))
    if not ppm_files:
        print(f"No .ppm files found in: {folder}")
        return

    for ppm_file in ppm_files:
        png_file = ppm_file.with_suffix(".png")
        img = Image.open(ppm_file)
        img.save(png_file)

    print(f"Converted {len(ppm_files)} files in {folder}")

convert_folder("F:/ai-design/dataset/gtsdb_yolo/images/train")
convert_folder("F:/ai-design/dataset/gtsdb_yolo/images/val")