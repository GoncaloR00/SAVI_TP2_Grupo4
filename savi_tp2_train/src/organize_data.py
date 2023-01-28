#!/usr/bin/python3

# This script's propose is to organize the data for training and testing a deep learning network

import os
import pathlib
from sklearn.model_selection import train_test_split
import shutil
from distutils.dir_util import copy_tree
import tqdm

data_path = pathlib.Path('../../../rgbd-dataset/')
reorg_path = pathlib.Path('../../../rgbd-dataset-custom/')
train_dir = pathlib.Path('../../../rgbd-dataset-train/')
test_dir = pathlib.Path('../../../rgbd-dataset-test/')

def main():
    if train_dir.is_dir() or test_dir.is_dir():
        print(f"There are already a train or a test set directory! Exiting")
        exit()

    # Reorganize Data
    if data_path.is_dir() or reorg_path.is_dir():
        print(f"Data directory exists! Continuing...")
    else:
        print(f"No data directory found! Aborting...")
        exit()

    if reorg_path.is_dir():
        print(f"{reorg_path} directory exists! Continuing...")
    else:
        print(f"Did not find {reorg_path} directory, creating one...")
        reorg_path.mkdir(parents=True, exist_ok=True)
        print(f"Reorganizing data...")
        directory_list = os.listdir(data_path)
        for directory_name in tqdm.tqdm(directory_list):
            dir_start = data_path / directory_name
            dir_end = reorg_path / directory_name
            dir_start_list = os.listdir(dir_start)
            for start_name in dir_start_list:
                copy_tree(str(dir_start / start_name), str(dir_end))

    print(f"Dividing between train (80%) and test data (20%)")
    dirs_list = []
    for root, _, files in os.walk(reorg_path, topdown=False):
        for name in files:
            if "_crop.png" in name:
                dirs_list.append(os.path.join(root, name))
    # print(dirs_list)
    train_image_filenames, test_image_filenames = train_test_split(dirs_list, test_size=0.2)

    print(f"Creating train folder and copying files to it")
    train_dir.mkdir(parents=True, exist_ok=True)
    for image in tqdm.tqdm(train_image_filenames):
        filepath = pathlib.Path(image)
        actual_dir = train_dir / filepath.parent.name
        if ~actual_dir.is_dir():
            actual_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(filepath, actual_dir / filepath.name)

    print(f"Creating test folder and copying files to it")
    test_dir.mkdir(parents=True, exist_ok=True)
    for image in tqdm.tqdm(test_image_filenames):
        filepath = pathlib.Path(image)
        actual_dir = test_dir / filepath.parent.name
        if ~actual_dir.is_dir():
            actual_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(filepath, actual_dir / filepath.name)

if __name__ == "__main__":
    main()
    print(f"Finished!")