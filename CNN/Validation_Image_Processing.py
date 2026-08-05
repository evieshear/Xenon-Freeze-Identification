from PIL import Image
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from tqdm import tqdm
import re


dir_val = r"C:\Users\joshy\OneDrive\Documents\Kravitz Lab\freezeData\0302 X"

def focus(filename):
    match = re.search(r"-f([\d]+(?:\.[\d]+)?)", filename)
    focus = float(match.group(1)) if match else -1
    return(focus)

def process_image(dir):
    output_dir = r"C:\Users\joshy\OneDrive\Documents\Kravitz Lab\freezeData\CNN Data\0302\\"
    crop_area = (450, 800, 850, 1200)
    img = Image.open(dir)
    cropped_img = img.crop(crop_area)
    cropped_img.save(output_dir + os.path.basename(dir))

# process_image(r"C:\Users\joshy\OneDrive\Documents\Kravitz Lab\RQs\liquid.png", "egg.png")

filelist = [
    os.path.join(dir_val, f) 
    for f in os.listdir(dir_val)
    if focus(f) == 4
] 

with ThreadPoolExecutor(max_workers=10) as executor:

    futures = [
        executor.submit(process_image, dir)
        for dir in filelist
    ]

    for future in tqdm(
        as_completed(futures),
        total=len(futures),
        desc="Processing images"
    ):
        result = future.result()