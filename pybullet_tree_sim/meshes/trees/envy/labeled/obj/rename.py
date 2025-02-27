#!/usr/bin/env python3
import glob
from pathlib import Path

files = glob.glob('*.obj')

for file in files:
    file_to_rename = Path(file)
    filename = file_to_rename.stem
    split = filename.split('_')
    split[-1] = split[-1].replace("tree", "")
    split[-1] = split[-1].zfill(5)
    
    new_name = '_'.join(split)+'.obj'
    
    file_to_rename.rename(new_name)