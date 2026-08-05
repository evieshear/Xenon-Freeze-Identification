from PIL import Image
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from tqdm import tqdm
import re

# Open the image
# img1 = Image.open(r"C:\Users\joshy\OneDrive\Documents\Kravitz Lab\freezeData\0309\20260307120236-b94.91-f4.png")
# img2 = Image.open(r"C:\Users\joshy\OneDrive\Documents\Kravitz Lab\freezeData\0309\20260307120042-b94.91-f4.png")


# Crop and save
# cropped_img1 = img1.crop(crop_area)
# cropped_img1.save("cropped_pillow1.jpg")

# cropped_img2 = img2.crop(crop_area)
# cropped_img2.save("cropped_pillow2.jpg")

dirs = [
    r"C:\Users\joshy\OneDrive\Documents\Kravitz Lab\freezeData\0309",
    r"C:\Users\joshy\OneDrive\Documents\Kravitz Lab\freezeData\0318",
    r"C:\Users\joshy\OneDrive\Documents\Kravitz Lab\freezeData\0322",
    r"C:\Users\joshy\OneDrive\Documents\Kravitz Lab\freezeData\0214",
    r"C:\Users\joshy\OneDrive\Documents\Kravitz Lab\freezeData\0506"
]

def focus(filename):
    match = re.search(r"-f([\d]+(?:\.[\d]+)?)", filename)
    focus = float(match.group(1)) if match else -1
    return(focus)

def process_image(dir, folder):
    output_dir = r"C:\Users\joshy\OneDrive\Documents\Kravitz Lab\freezeData\CNN Data\\"
    crop_area = (450, 800, 850, 1200)
    img = Image.open(dir)
    cropped_img = img.crop(crop_area)
    cropped_img.save(output_dir + folder + r'\\' + os.path.basename(dir))

# process_image(r"C:\Users\joshy\OneDrive\Documents\Kravitz Lab\RQs\liquid.png", "egg.png")

def list_files(dir):
    return([
        (os.path.join(dir, f), Path(dir).name) 
        for f in os.listdir(dir)
        if focus(f) == 4
    ])

files_and_paths = []


for i in range(len(dirs)):
    files_and_paths.append([])
    files_and_paths[i] = list_files(dirs[i])

with ThreadPoolExecutor(max_workers=10) as executor:

    futures = [
        executor.submit(process_image, dir, folder)
        for files_for_given_path in files_and_paths
        for dir, folder in files_for_given_path
    ]

    for future in tqdm(
        as_completed(futures),
        total=len(futures),
        desc="Processing images"
    ):
        result = future.result()