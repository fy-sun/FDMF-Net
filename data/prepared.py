from skimage import io
import os
import random
from pathlib import Path
import rasterio
from rasterio.windows import Window
import numpy as np
from tqdm import tqdm

# 配置参数
class Config:
    input_dir = Path('./')  # replace with actual path
    output_dir = Path('./OPT_SAR_WHU_crops')
    modalities = ['GT', 'OPT', 'SAR']
    patch_size = 256
    patches_per_image = 100
    train_ratio = 0.8
    test_ratio = 0.2
    random_seed = 42

random.seed(Config.random_seed)
np.random.seed(Config.random_seed)

def create_output_dirs(config):
    for split in ['train', 'test']:
        for modality in config.modalities:
            dir_path = config.output_dir / split / modality
            dir_path.mkdir(parents=True, exist_ok=True)

def get_all_filenames(config):
    gt_dir = config.input_dir / 'GT'
    filenames = sorted([f.name for f in gt_dir.glob('*.tif')])
    return filenames

def generate_non_overlapping_windows(width, height, patch_size, patches_required):
    max_x = width // patch_size
    max_y = height // patch_size
    total_patches = max_x * max_y

    if total_patches < patches_required:
        raise ValueError(f"Image Size is Small, Need: {patches_required}, Max: {total_patches}")

    all_windows = [(i * patch_size, j * patch_size) for i in range(max_x) for j in range(max_y)]
    random.shuffle(all_windows)
    selected_windows = all_windows[:patches_required]
    return selected_windows

def process_and_save_patches(filename, config, windows):
    images = {}
    for modality in config.modalities:
        image_path = config.input_dir / modality / filename
        with rasterio.open(image_path) as src:
            images[modality] = src.read()
            width = src.width
            height = src.height

    try:
        selected_windows = generate_non_overlapping_windows(width, height, config.patch_size, config.patches_per_image)
    except ValueError as e:
        print(f"Skipping {filename}: {e}")
        return

    for idx, (x, y) in enumerate(selected_windows):
        window = Window(x, y, config.patch_size, config.patch_size)

        cropped = {}
        for modality in config.modalities:
            cropped[modality] = images[modality][:, y:y+config.patch_size, x:x+config.patch_size]

        rnd = random.random()
        if rnd < config.train_ratio:
            split = 'train'
        else:
            split = 'test'

        patch_filename = f"{Path(filename).stem}_patch_{idx+1:03d}.tif"

        for modality in config.modalities:
            output_path = config.output_dir / split / modality / patch_filename
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=config.patch_size,
                width=config.patch_size,
                count=images[modality].shape[0],
                dtype=images[modality].dtype,
                crs=None,
                transform=None
            ) as dst:
                dst.write(cropped[modality])


def main():
    config = Config()

    create_output_dirs(config)
    print(f"{config.output_dir}")

    filenames = get_all_filenames(config)
    print(f"{len(filenames)}")

    for filename in tqdm(filenames, desc="Processing images"):
        process_and_save_patches(filename, config, windows=None)

    print("Finish! ")


if __name__ == '__main__':
    main()

