'''
For licensing see accompanying LICENSE.txt file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
'''
import os
import h5py
import numpy as np
import fnmatch
from tqdm import tqdm

def index_episodes(dataset_path): 
    # find all hdf5 files
    hdf5_files = []
    for root, dirs, files in os.walk(dataset_path):
        for filename in fnmatch.filter(files, "*.hdf5"):
            hdf5_files.append(os.path.join(root, filename))
            # break # debug
        # break # debug
    print(f"Found {len(hdf5_files)} hdf5 files")

    # get lengths of all hdf5 files
    all_episode_len = []
    for dataset_path in tqdm(hdf5_files, desc='iterating dataset_path to get all episode lengths...'):
        try:
            with h5py.File(dataset_path, "r") as root:
                action = root['/transforms/leftHand'][()]
        except Exception as e:
            print(f"Error loading {dataset_path}")
        all_episode_len.append(len(action))
    
    return hdf5_files, all_episode_len

def get_camera_intrinsics():
    # utility function to get camera intrinsics, if not getting it from the data. it is always the same.
    return np.array([[736.6339, 0., 960.], [0., 736.6339, 540.], [0., 0., 1.]], dtype=np.float32)

def convert_to_camera_frame(tfs, cam_ext):
    '''
    convert transforms from the world frame to the camera frame
    tfs: a set of transforms in the world frame, shape N x 4 x 4
    cam_ext: camera extrinsics in the world frame, shape 4 x 4
    '''
    return np.linalg.inv(cam_ext)[None] @ tfs
