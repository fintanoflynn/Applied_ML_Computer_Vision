"""
Download the PlantVillage dataset and place it inside data/raw/
"""

from pathlib import Path
import shutil
import kagglehub

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

DATASET_HANDLE = "abdallahalidev/plantvillage-dataset"

def find_image_root(path: Path) -> Path:
    """Return the directory whose immediate children are the class folders.

    kagglehub may return a wrapper directory (e.g. .../plantvillage dataset/color).
    Walk down single-child folders to find the first one whose children look like
    class directories (each containing image files).
    """
    current = path
    for _ in range(6):
        children = [c for c in current.iterdir() if c.is_dir() and not c.name.startswith(".")]
        if not children:
            break
        # If any child contains an image, this is the class-root directory.
        for child in children:
            if any(p.suffix.lower() in {".jpg", ".jpeg", ".png"} for p in child.iterdir() if p.is_file()):
                return current
        # Prefer a "color" subfolder if present (PlantVillage has color/grayscale/segmented).
        color = next((c for c in children if c.name.lower() == "color"), None)
        current = color if color is not None else children[0]
    raise FileNotFoundError("Could not find the PlantVillage image root.")


def main() -> None:

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading dataset...")
    downloaded_path = Path(kagglehub.dataset_download(DATASET_HANDLE))

    print(f"Dataset downloaded to: {downloaded_path}")

    class_root = find_image_root(downloaded_path)
    print(f"Found class folders at: {class_root}")

    for class_folder in class_root.iterdir():
        if class_folder.is_dir():
            destination = RAW_DIR / class_folder.name

            if destination.exists():
                print(f"Skipping existing folder: {destination.name}")
                continue

            shutil.copytree(class_folder, destination)
            print(f"Copied: {class_folder.name}")

    print(f"\nDone. Raw data is now in: {RAW_DIR}")


if __name__ == "__main__":
    main()
