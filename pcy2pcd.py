#!/usr/bin/env python3

import pathlib
import os

original_path = pathlib.Path('../rgbd-scenes-v2/pc')
final_path = pathlib.Path('../rgbd-scenes-v2/pcd')

final_path.mkdir(parents=True, exist_ok=True)

for _, _, files in os.walk(original_path, topdown=False):
        for name in files:
            if ".ply" in name:
                ipath = original_path / name
                fname = name.split('.')[0] + '.pcd'
                fpath = final_path / fname
                os.system('pcl_ply2pcd ' + str(ipath) + ' ' + str(fpath))