import math
import numpy as np
from PIL import Image
from pathlib import Path
import os

script_dir = Path(__file__).parent.resolve()

def conv_output_size(input_size, padding, kernel_size, stride):
    return math.floor((input_size + 2 * padding - kernel_size)/stride + 1)

def get_image(filename: str):
    array = np.load(filename)
    return array

def show_image(array):
    unnormalized_array = (array * 255).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(unnormalized_array)
    image.show()

def save_image(filename: str):
    array = get_image(filename)
    unnormalized_array = (((array + 1) / 2) * 255).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(unnormalized_array)
    image.save(f'{filename}.png')

def get_filenames(dir: str, ext: str = ''):
    directory = Path(dir)
    files = [f.name for f in directory.iterdir() if f.is_file() and f.suffix == ext]
    return np.array(files)

def get_images_from_file(dir: str):
    images = []
    filenames: list[str] = get_filenames(dir)
    for file in filenames:
        images.append(get_image(script_dir / dir / file))
    return np.array(images)