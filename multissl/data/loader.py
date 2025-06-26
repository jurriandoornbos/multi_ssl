# Copyright 2025 Jurrian Doornbos
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0


import tifffile as tiff
import os
from PIL import Image
import numpy as np

def tifffile_loader(f):
    img_path = f
    
    # Check file extension first
    _, ext = os.path.splitext(img_path.lower())
    
    try:
        if ext in ['.tif', '.tiff']:
            with tiff.TiffFile(img_path) as tif:
                image_array = tif.asarray()
        else:
            # Use PIL for other formats (JPEG, PNG, etc.)
            with Image.open(img_path) as img:
                image_array = np.array(img)
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        return None
   
    return image_array
