# Copyright 2025 Jurrian Doornbos
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0


import tifffile as tiff
#from PIL import Image

def tifffile_loader(f):
    img_path = f
    with tiff.TiffFile(img_path) as tif:
        image_array = tif.asarray()
    
    return image_array
