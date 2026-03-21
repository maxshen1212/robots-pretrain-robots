# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Data Loaders for VisualEncoder Training

This module provides data loaders for different robot datasets including:
- EPIC: Human demonstration dataset with language instructions
- DROID: Robot demonstration dataset
- Ego4D: First-person video dataset

Each loader supports temporal sampling, data augmentation, and various data formats.
"""

import warnings
import pdb
import torchvision
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision.transforms.functional')

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import IterableDataset
import pandas as pd
import json
import time
from tqdm import tqdm
import pickle
from torchvision.utils import save_image
import json
import random
from scipy.spatial.transform import Rotation
import h5py

def matrix_to_rotation_6d(mat: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to 6D rotation representation.
    
    The 6D representation uses the first two columns of the rotation matrix,
    which is more stable for learning than quaternions or Euler angles.
    
    Args:
        mat: Rotation matrix of shape [..., 3, 3]
        
    Returns:
        6D rotation representation of shape [..., 6]
    """
    return mat[:, :2].reshape(-1)


def quat_xyzw_to_rot6d(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion (xyzw format) to 6D rotation representation.
    
    Args:
        quat: Quaternion in xyzw format
        
    Returns:
        6D rotation representation
    """
    return matrix_to_rotation_6d(Rotation.from_quat(quat).as_matrix())

def rotvec_to_rot6d(rotvec: np.ndarray) -> np.ndarray:
    """
    Convert rotation vector to 6D rotation representation.
    
    Args:
        rotvec: Rotation vector
        
    Returns:
        6D rotation representation
    """
    return matrix_to_rotation_6d(Rotation.from_rotvec(rotvec).as_matrix())


def build_eef_vec(pos: np.ndarray, rot6d: np.ndarray, grip: float) -> np.ndarray:
    """
    Build end-effector vector from position, rotation, and gripper state.
    
    Args:
        pos: Position vector [x, y, z]
        rot6d: 6D rotation representation
        grip: Gripper state (0=closed, 1=open)
        
    Returns:
        Concatenated end-effector vector
    """
    return np.concatenate([pos, rot6d, [grip]]).astype(np.float32)

def get_ind(vid, index, ds):
    """
    Load image from video directory based on dataset type.
    
    Args:
        vid: Video directory path
        index: Frame index
        ds: Dataset type ("ego4d" or "droid")
        
    Returns:
        Loaded image tensor
    """
    if ds == "ego4d":
        return torchvision.io.read_image(f"{vid}/{index:06}.jpg")
    elif ds == 'droid':
        return torchvision.io.read_image(f"{vid}/{index}.png", mode=torchvision.io.image.ImageReadMode.RGB)
    else:
        raise NameError('Invalid Dataset')


## Data Loader for Ego4D
class VisualEncoderBuffer(IterableDataset):
    """
    Data loader for Ego4D dataset.
    
    This loader samples temporal sequences from Ego4D videos for training
    temporal contrastive learning and language alignment tasks.
    
    Features:
    - Temporal sampling with configurable alpha parameter
    - Data augmentation support
    - Language instruction extraction
    - Multi-view support
    """
    
    def __init__(self, ego4dpath, num_workers, source1, source2, alpha, datasources, doaug = "none"):
        """
        Initialize Ego4D data loader.
        
        Args:
            ego4dpath: Path to Ego4D dataset
            num_workers: Number of data loading workers
            source1, source2: Source identifiers (legacy parameters)
            alpha: Temporal sampling parameter (0-1)
            datasources: List of data sources to use
            doaug: Data augmentation type ("none", "rc", "rctraj")
        """
        self._num_workers = max(1, num_workers)
        self.alpha = alpha
        self.curr_same = 0
        self.data_sources = datasources
        self.doaug = doaug

        # Augmentations
        if doaug in ["rc", "rctraj"]:
            self.aug = torch.nn.Sequential(
                transforms.RandomResizedCrop(224, scale = (0.2, 1.0)),
            )
        else:
            self.aug = lambda a : a

        # Load Data
        if "ego4d" in self.data_sources:
            print("Ego4D")
            self.manifest = pd.read_csv(f"{ego4dpath}manifest.csv")
            print(self.manifest)
            self.ego4dlen = len(self.manifest)
        else:
            raise NameError('Invalid Dataset')


    def _sample(self):
        """
        Sample a single training example from the dataset.
        
        Returns:
            Tuple of (image sequence, language instruction)
        """
        t0 = time.time()
        ds = random.choice(self.data_sources)

        vidid = np.random.randint(0, self.ego4dlen)
        m = self.manifest.iloc[vidid]
        vidlen = m["len"]
        txt = m["txt"]
        label = txt[2:] ## Cuts of the "C " part of the text
        vid = m["path"]

        start_ind = np.random.randint(1, 2 + int(self.alpha * vidlen))
        end_ind = np.random.randint(int((1-self.alpha) * vidlen)-1, vidlen)
        s1_ind = np.random.randint(2, vidlen)
        s0_ind = np.random.randint(1, s1_ind)
        s2_ind = np.random.randint(s1_ind, vidlen+1) # start, s0, s1, s2, end

        if self.doaug == "rctraj":
            ### Encode each image in the video at once the same way
            im0 = get_ind(vid, start_ind, ds) 
            img = get_ind(vid, end_ind, ds)
            imts0 = get_ind(vid, s0_ind, ds)
            imts1 = get_ind(vid, s1_ind, ds)
            imts2 = get_ind(vid, s2_ind, ds)
            allims = torch.stack([im0, img, imts0, imts1, imts2], 0)
            allims_aug = self.aug(allims / 255.0) * 255.0

            im0 = allims_aug[0]
            img = allims_aug[1]
            imts0 = allims_aug[2]
            imts1 = allims_aug[3]
            imts2 = allims_aug[4]
        else:
            ### Encode each image individually
            im0 = self.aug(get_ind(vid, start_ind, ds) / 255.0) * 255.0
            img = self.aug(get_ind(vid, end_ind, ds) / 255.0) * 255.0
            imts0 = self.aug(get_ind(vid, s0_ind, ds) / 255.0) * 255.0
            imts1 = self.aug(get_ind(vid, s1_ind, ds) / 255.0) * 255.0
            imts2 = self.aug(get_ind(vid, s2_ind, ds) / 255.0) * 255.0

        im = torch.stack([im0, img, imts0, imts1, imts2])
        return (im, label)

    def __iter__(self):
        """Return infinite iterator over dataset."""
        while True:
            yield self._sample()

## Data Loader for Droid
class VisualEncoderBufferDroid(IterableDataset):
    """
    Data loader for DROID dataset.
    
    This loader handles robot demonstration data with proprioceptive information,
    language instructions, and multi-view camera data.
    
    Features:
    - Multi-view camera support
    - Proprioceptive state data
    - Language instruction processing
    - Temporal sampling for contrastive learning
    """
    
    def __init__(self, droidpath, num_workers, source1, source2, alpha,
                    datasources, doaug = "none", state_list_used = None, state_window=1, use_action=False, view_keys_used = None):
        """
        Initialize DROID data loader.
        
        Args:
            droidpath: Path to DROID dataset
            num_workers: Number of data loading workers
            source1, source2: Source identifiers (legacy parameters)
            alpha: Temporal sampling parameter
            datasources: List of data sources to use
            doaug: Data augmentation type
            state_list_used: List of state keys to use
            state_window: Number of timesteps for state encoding
            use_action: Whether to include action data
            view_keys_used: List of camera view keys to use
        """
        self._num_workers = max(1, num_workers)
        self.alpha = alpha
        self.curr_same = 0
        self.data_sources = datasources
        self.doaug = doaug
        self.dataset_path = droidpath
        self.state_keys = ['cartesian_position', 'gripper_position', 'joint_position']
        self.lang_keys = ['language_instruction', 'language_instruction_2', 'language_instruction_3']
        self.view_keys = view_keys_used # ['exterior_image_1_left', 'exterior_image_2_left', 'wrist_image_left']
        self.state_list_used = state_list_used
        self.state_window = state_window
        self.use_action = use_action

        # Augmentations
        if doaug in ["rc", "rctraj"]:
            self.aug = torch.nn.Sequential(
                transforms.RandomResizedCrop(224, scale = (0.5, 1.0)), # first crop, then resize
            )
        elif doaug in ["rctraj_eval"]:
            self.aug = torch.nn.Sequential(
                transforms.Resize(256),
                transforms.CenterCrop(224),
            )
        else:
            self.aug = lambda a : a

        # Load Data
        if "droid" in self.data_sources:
            print("Droid")
            self.loaded_dataset = os.listdir(droidpath)
            print(self.loaded_dataset[:5])
            self.datasetlen = len(self.loaded_dataset)
        else:
            raise NameError('Invalid Dataset')


    def _sample(self):
        """
        Sample a single training example from the DROID dataset.
        
        Returns:
            Tuple containing images, language, states, actions, and metadata
        """
        t0 = time.time()
        ds = random.choice(self.data_sources)

        vidid = np.random.randint(0, self.datasetlen)
        traj_path = self.loaded_dataset[vidid] # 2023-02-28_Tue_Feb_28_20:31:42_2023
        vidlen = min(len(os.listdir(os.path.join(self.dataset_path, traj_path, 'exterior_image_1_left'))), len(os.listdir(os.path.join(self.dataset_path, traj_path, 'exterior_image_2_left'))))
        txt_path = random.choice(self.lang_keys)
        with open(os.path.join(self.dataset_path, traj_path, txt_path, '0.txt'), 'r') as file:
            label = file.read()
        vid_path = random.choice(self.view_keys) # time contrastive within same view
        vid = os.path.join(self.dataset_path, traj_path, vid_path) # video path
        otherdata_path = os.path.join(self.dataset_path, traj_path, 'other_data.pkl')

        start_ind = np.random.randint(1, 2 + int(self.alpha * vidlen)) # [low, high)
        end_ind = np.random.randint(int((1-self.alpha) * vidlen)-1, vidlen)
        s1_ind = np.random.randint(2, vidlen)
        s0_ind = np.random.randint(1, s1_ind)
        s2_ind = np.random.randint(s1_ind, vidlen) # start, s0, s1, s2, end

        # for state encode
        with open(otherdata_path, 'rb') as f:
            loaded_data = pickle.load(f)
        state_array, full_state_dict = np.empty(0), {'s0': np.empty(0), 's2': np.empty(0)}
        for key in self.state_list_used:
            state_array = np.concatenate((state_array, loaded_data[key][s0_ind]))

        s0wind_start = max(1, s0_ind - self.state_window // 2)
        s2wind_start = max(1, s2_ind - self.state_window // 2)
        for i in range(self.state_window):
            for key in self.state_keys:
                full_state_dict['s0'] = np.concatenate((full_state_dict['s0'], loaded_data[key][min(s0wind_start + i, vidlen - 1)]))
                full_state_dict['s2'] = np.concatenate((full_state_dict['s2'], loaded_data[key][min(s2wind_start + i, vidlen - 1)]))
            if self.use_action and i != self.state_window - 1:
                full_state_dict['s0'] = np.concatenate((full_state_dict['s0'], loaded_data['action'][min(s0wind_start + i, vidlen - 1)]))
                full_state_dict['s2'] = np.concatenate((full_state_dict['s2'], loaded_data['action'][min(s2wind_start + i, vidlen - 1)]))

        full_state_dict['s0'] = torch.tensor(full_state_dict['s0']).float()
        full_state_dict['s2'] = torch.tensor(full_state_dict['s2']).float()

        # for bc, sample action
        actions = torch.tensor(np.stack([loaded_data['action'][start_ind], 
                                loaded_data['action'][end_ind], 
                                loaded_data['action'][s0_ind], 
                                loaded_data['action'][s1_ind], 
                                loaded_data['action'][s2_ind]])).float()
        # actions = torch.tensor(loaded_data['action'][s0_ind]).float()

        if self.doaug == ["rctraj", "rctraj_eval"]:
            ### Encode each image in the video at once the same way
            im0 = get_ind(vid, start_ind, ds)
            img = get_ind(vid, end_ind, ds)
            imts0 = get_ind(vid, s0_ind, ds)
            imts1 = get_ind(vid, s1_ind, ds)
            imts2 = get_ind(vid, s2_ind, ds)
            allims = torch.stack([im0, img, imts0, imts1, imts2], 0)
            allims_aug = self.aug(allims / 255.0) * 255.0

            im0 = allims_aug[0]
            img = allims_aug[1]
            imts0 = allims_aug[2]
            imts1 = allims_aug[3]
            imts2 = allims_aug[4]
        else:
            ### Encode each image individually
            im0 = self.aug(get_ind(vid, start_ind, ds) / 255.0) * 255.0
            img = self.aug(get_ind(vid, end_ind, ds) / 255.0) * 255.0
            imts0 = self.aug(get_ind(vid, s0_ind, ds) / 255.0) * 255.0
            imts1 = self.aug(get_ind(vid, s1_ind, ds) / 255.0) * 255.0
            imts2 = self.aug(get_ind(vid, s2_ind, ds) / 255.0) * 255.0

        im = torch.stack([im0, img, imts0, imts1, imts2])
        return (im, label, torch.tensor(state_array).float(), full_state_dict, actions)

    def __iter__(self):
        while True:
            yield self._sample()


ACTION_DIM = 24  # Each action has 24 values (12 for left, 12 for right)
ACTION_SAMPLING_STRIDE = 4  # Sample every 4th action from 32 total actions
NUM_ACTIONS = 8  # Number of actions to sample (32 // 4)

# Constants for normalization
MAX_X_PIXEL_VALUE = 456.0  # Maximum pixel value for action normalization
MAX_Y_PIXEL_VALUE = 256.0  # Maximum pixel value for action normalization

def get_action_2d_rot_grip_indices():
    """
    Create indices for extracting 2D pixel actions and rotation/gripper components.
    
    Each action originally has 24 values: (x, y, z, x2d, y2d, r1, r2, r3, r4, r5, r6, g)
    We remove the xyz (3D) actions, leaving 18 values: (x2d, y2d, r1, r2, r3, r4, r5, r6, g)
    
    Returns:
        List of indices for extracting 2D and rotation/gripper components
    """
    indices = []
    for act_2d_i in range(0, NUM_ACTIONS*ACTION_SAMPLING_STRIDE, ACTION_SAMPLING_STRIDE):
        base_idx = act_2d_i * ACTION_DIM
        # Indices for 2D actions and rotation/gripper (skip xyz: indices 0,1,2)
        # Left hand: x2d(3), y2d(4), r1(5), r2(6), r3(7), r4(8), r5(9), r6(10), g(11)
        # Right hand: x2d(15), y2d(16), r1(17), r2(18), r3(19), r4(20), r5(21), r6(22), g(23)
        left_indices = [base_idx + i for i in [3, 4, 5, 6, 7, 8, 9, 10, 11]]
        right_indices = [base_idx + i for i in [15, 16, 17, 18, 19, 20, 21, 22, 23]]
        indices.extend(left_indices + right_indices)
    return indices


def get_action_2d_pixel_indices():
    indices = []
    for act_2d_i in range(0, NUM_ACTIONS*ACTION_SAMPLING_STRIDE, ACTION_SAMPLING_STRIDE):
        base_idx = act_2d_i * ACTION_DIM
        # Indices for 2D actions
        # Left hand: x2d(3), y2d(4)
        # Right hand: x2d(15), y2d(16)
        indices.extend([base_idx + i for i in [3, 4, 15, 16]])
    return indices

def get_action_2d_pixel_x_indices():
    indices = []
    for act_2d_i in range(0, NUM_ACTIONS*ACTION_SAMPLING_STRIDE, ACTION_SAMPLING_STRIDE):
        base_idx = act_2d_i * ACTION_DIM
        indices.extend([base_idx + i for i in [3, 15]])
    return indices

def get_action_2d_pixel_y_indices():
    indices = []
    for act_2d_i in range(0, NUM_ACTIONS*ACTION_SAMPLING_STRIDE, ACTION_SAMPLING_STRIDE):
        base_idx = act_2d_i * ACTION_DIM
        indices.extend([base_idx + i for i in [4, 16]])
    return indices

def get_action_2d_pixel_x_left_indices():
    indices = []
    for act_2d_i in range(0, NUM_ACTIONS*ACTION_SAMPLING_STRIDE, ACTION_SAMPLING_STRIDE):
        base_idx = act_2d_i * ACTION_DIM
        indices.extend([base_idx + i for i in [3]])
    return indices

def get_action_2d_pixel_y_left_indices():
    indices = []
    for act_2d_i in range(0, NUM_ACTIONS*ACTION_SAMPLING_STRIDE, ACTION_SAMPLING_STRIDE):
        base_idx = act_2d_i * ACTION_DIM
        indices.extend([base_idx + i for i in [4]])
    return indices


def get_action_2d_pixel_x_right_indices():
    indices = []
    for act_2d_i in range(0, NUM_ACTIONS*ACTION_SAMPLING_STRIDE, ACTION_SAMPLING_STRIDE):
        base_idx = act_2d_i * ACTION_DIM
        indices.extend([base_idx + i for i in [15]])
    return indices

def get_action_2d_pixel_y_right_indices():
    indices = []
    for act_2d_i in range(0, NUM_ACTIONS*ACTION_SAMPLING_STRIDE, ACTION_SAMPLING_STRIDE):
        base_idx = act_2d_i * ACTION_DIM
        indices.extend([base_idx + i for i in [16]])
    return indices


class BaseBufferEpicH5(IterableDataset):
    def __init__(self, dataset_path, num_workers, split, 
                 datasources, doaug="none", include_rot_grip=False, center_crop=False):
        self._num_workers = max(1, num_workers)
        self.curr_same = 0
        self.data_sources = datasources
        self.doaug = doaug
        self.dataset_path = Path(dataset_path)
        self.include_rot_grip = include_rot_grip
        self.split = split
        self.split_ratio = 0.9
        self.random_seed = 42
        self.center_crop = center_crop

        # Augmentations
        if doaug in ["resize"]:
            self.aug = torch.nn.Sequential(
                transforms.Resize((240, 240)),
            )
        elif doaug in ["resize_vit"]:
            self.aug = torch.nn.Sequential(
                transforms.Resize((240, 240)),
                transforms.CenterCrop((224, 224)),
            )
        else:
            self.aug = lambda a: a

        # self._init_h5file()
        # self._sample()
        # self._get_actions_mask()
        self._init_stats()
    
    def _init_stats(self):
        with h5py.File(self.dataset_path, 'r', libver='latest', swmr=True) as f:
            self.all_demo_keys = list(sorted(f['data'].keys()))

            self.actions_mean, self.actions_std = self._get_actions_mean_std(f)
            self.contact_gmm_left_mean, self.contact_gmm_left_std = self._get_contact_gmm_left_mean_std(f)
            self.contact_gmm_right_mean, self.contact_gmm_right_std = self._get_contact_gmm_right_mean_std(f)
            self.obj_info_left_mean, self.obj_info_left_std = self._get_obj_info_left_mean_std(f)
            self.obj_info_right_mean, self.obj_info_right_std = self._get_obj_info_right_mean_std(f)

            self.stats = {
                "actions_mean": self.actions_mean,
                "actions_std": self.actions_std,
                "contact_gmm_left_mean": self.contact_gmm_left_mean,
                "contact_gmm_left_std": self.contact_gmm_left_std,
                "contact_gmm_right_mean": self.contact_gmm_right_mean,
                "contact_gmm_right_std": self.contact_gmm_right_std,
                "obj_info_left_mean": self.obj_info_left_mean,
                "obj_info_left_std": self.obj_info_left_std,
                "obj_info_right_mean": self.obj_info_right_mean,
                "obj_info_right_std": self.obj_info_right_std,
            }

            self._get_actions_mask(f)
            self._get_contact_left_mask(f)
            self._get_contact_right_mask(f)
            self._get_obj_left_mask(f)
            self._get_obj_right_mask(f)
            
    
    def _init_h5file(self):
        # This is called inside __iter__, so each worker gets its own handle
        self.h5file = h5py.File(self.dataset_path, 'r', libver='latest', swmr=True)

        all_demo_keys = sorted(self.h5file['data'].keys())

        # Deterministic shuffling
        rng = np.random.default_rng(self.random_seed)
        all_demo_keys = list(all_demo_keys)  # Ensure it's a mutable list
        rng.shuffle(all_demo_keys)

        num_total = len(all_demo_keys)
        num_train = int(self.split_ratio * num_total)

        if self.split == "train":
            self.demo_keys = all_demo_keys[:num_train]
        elif self.split == "val":
            self.demo_keys = all_demo_keys[num_train:]
        else:
            raise ValueError(f"Unknown split: {self.split}")

        self.datasetlen = len(self.demo_keys)

    def _get_actions_mask(self, f):
        min_bound = (240 - 224) / 2
        max_bound = 240 - min_bound
        self.actions_mask_left = {}
        self.actions_mask_right = {}
        for demo_key in self.all_demo_keys:
            actions = f['data'][demo_key]['action']
            action_2d_pixel_x_left_indices = get_action_2d_pixel_x_left_indices()
            action_2d_pixel_y_left_indices = get_action_2d_pixel_y_left_indices()
            action_2d_pixel_x_right_indices = get_action_2d_pixel_x_right_indices()
            action_2d_pixel_y_right_indices = get_action_2d_pixel_y_right_indices()
            x_actions_left = actions[:,action_2d_pixel_x_left_indices] * 240 / 456.0
            y_actions_left = actions[:,action_2d_pixel_y_left_indices] * 240 / 256.0
            x_actions_right = actions[:,action_2d_pixel_x_right_indices] * 240 / 456.0
            y_actions_right = actions[:,action_2d_pixel_y_right_indices] * 240 / 256.0

            mask_left = torch.ones(actions.shape[0])
            mask_right = torch.ones(actions.shape[0])
            for i in range(actions.shape[0]):
                if x_actions_left[i].min() < min_bound or x_actions_left[i].max() > max_bound or y_actions_left[i].min() < min_bound or y_actions_left[i].max() > max_bound:
                    mask_left[i] = 0
                if x_actions_right[i].min() < min_bound or x_actions_right[i].max() > max_bound or y_actions_right[i].min() < min_bound or y_actions_right[i].max() > max_bound:
                    mask_right[i] = 0

            self.actions_mask_left[demo_key] = mask_left
            self.actions_mask_right[demo_key] = mask_right

    def _get_actions_mean_std(self, f):
        # Get the mean and std of the dataset
        all_actions = []
        for demo_key in self.all_demo_keys:
            actions = f['data'][demo_key]['action']
            all_actions.append(actions)

        all_actions = np.concatenate(all_actions, axis=0)
        mean = np.mean(all_actions, axis=0)
        std = np.std(all_actions, axis=0)
        return mean, std
    
    def _get_contact_left_mask(self, f):
        min_bound = (240 - 224) / 2
        max_bound = 240 - min_bound
        self.contact_left_mask = {}
        for demo_key in self.all_demo_keys:
            contact_gmm_left = f['data'][demo_key]['contact/gmm_contacts_left']
            mask_left = torch.ones(contact_gmm_left.shape[0])
            for i in range(contact_gmm_left.shape[0]):
                contact = contact_gmm_left[i]
                contact_x = contact[:,0] * 240 / 456.0
                contact_y = contact[:,1] * 240 / 256.0
                if (contact_x < min_bound).any() or (contact_x > max_bound).any() or (contact_y < min_bound).any() or (contact_y > max_bound).any():
                    mask_left[i] = 0
            self.contact_left_mask[demo_key] = mask_left
            
    def _get_contact_gmm_left_mean_std(self, f):
        # Get the mean and std of the dataset
        all_contact_gmm_left = []
        for demo_key in self.all_demo_keys:
            contact_gmm_left = np.array(f['data'][demo_key]['contact/gmm_contacts_left'])
            y_values = contact_gmm_left[:, :, 0]
            sorted_indices = np.argsort(y_values, axis=1)
            sorted_indices_expanded = sorted_indices[:, :, np.newaxis]
            contact_gmm_left_sorted = np.take_along_axis(contact_gmm_left, sorted_indices_expanded, axis=1)
            all_contact_gmm_left.append(contact_gmm_left_sorted)
        all_contact_gmm_left = np.concatenate(all_contact_gmm_left, axis=0)
        mean = np.mean(all_contact_gmm_left, axis=0)
        std = np.std(all_contact_gmm_left, axis=0)
        return mean, std
    
    def _get_contact_right_mask(self, f):
        min_bound = (240 - 224) / 2
        max_bound = 240 - min_bound
        self.contact_right_mask = {}
        for demo_key in self.all_demo_keys:
            contact_gmm_right = f['data'][demo_key]['contact/gmm_contacts_right']
            mask_right = torch.ones(contact_gmm_right.shape[0])
            for i in range(contact_gmm_right.shape[0]):
                contact = contact_gmm_right[i]
                contact_x = contact[:,0] * 240 / 456.0
                contact_y = contact[:,1] * 240 / 256.0
                if (contact_x < min_bound).any() or (contact_x > max_bound).any() or (contact_y < min_bound).any() or (contact_y > max_bound).any():
                    mask_right[i] = 0
            self.contact_right_mask[demo_key] = mask_right
    
    def _get_contact_gmm_right_mean_std(self, f):
        # Get the mean and std of the dataset
        all_contact_gmm_right = []
        for demo_key in self.all_demo_keys:
            contact_gmm_right = np.array(f['data'][demo_key]['contact/gmm_contacts_right'])
            y_values = contact_gmm_right[:, :, 0]
            sorted_indices = np.argsort(y_values, axis=1)
            sorted_indices_expanded = sorted_indices[:, :, np.newaxis]
            contact_gmm_right_sorted = np.take_along_axis(contact_gmm_right, sorted_indices_expanded, axis=1)
            all_contact_gmm_right.append(contact_gmm_right_sorted)
        all_contact_gmm_right = np.concatenate(all_contact_gmm_right, axis=0)
        mean = np.mean(all_contact_gmm_right, axis=0)
        std = np.std(all_contact_gmm_right, axis=0)
        return mean, std
    
    def _get_obj_left_mask(self, f):
        min_bound = (240 - 224) / 2
        max_bound = 240 - min_bound
        self.obj_left_mask = {}
        for demo_key in self.all_demo_keys:
            obj_info = f['data'][demo_key]['contact/intersected_bbox_left']
            mask = torch.ones(obj_info.shape[0])
            for i in range(obj_info.shape[0]):
                x0 = obj_info[i][0] * 240 / 456.0
                y0 = obj_info[i][1] * 240 / 256.0
                x1 = obj_info[i][2] * 240 / 456.0
                y1 = obj_info[i][3] * 240 / 256.0
                if x0 < min_bound or y0 < min_bound or x1 > max_bound or y1 > max_bound:
                    mask[i] = 0
            self.obj_left_mask[demo_key] = mask

    def _get_obj_info_left_mean_std(self, f):
        # Get the mean and std of the dataset
        all_obj_info = []
        for demo_key in self.all_demo_keys:
            obj_info = f['data'][demo_key]['contact/intersected_bbox_left']
            all_obj_info.append(obj_info)
        all_obj_info = np.concatenate(all_obj_info, axis=0)
        mean = np.mean(all_obj_info, axis=0)
        std = np.std(all_obj_info, axis=0)
        return mean, std
    
    def _get_obj_right_mask(self, f):
        min_bound = (240 - 224) / 2
        max_bound = 240 - min_bound
        self.obj_right_mask = {}
        for demo_key in self.all_demo_keys:
            obj_info = f['data'][demo_key]['contact/intersected_bbox_right']
            mask = torch.ones(obj_info.shape[0])
            for i in range(obj_info.shape[0]):
                x0 = obj_info[i][0] * 240 / 456.0
                y0 = obj_info[i][1] * 240 / 256.0
                x1 = obj_info[i][2] * 240 / 456.0
                y1 = obj_info[i][3] * 240 / 256.0
                if x0 < min_bound or y0 < min_bound or x1 > max_bound or y1 > max_bound:
                    mask[i] = 0
            self.obj_right_mask[demo_key] = mask
    
    def _get_obj_info_right_mean_std(self, f):
        # Get the mean and std of the dataset
        all_obj_info = []
        for demo_key in self.all_demo_keys:
            obj_info = f['data'][demo_key]['contact/intersected_bbox_right']
            all_obj_info.append(obj_info)
        all_obj_info = np.concatenate(all_obj_info, axis=0)
        mean = np.mean(all_obj_info, axis=0)
        std = np.std(all_obj_info, axis=0)
        return mean, std

    def _sample(self):
        # If we haven't opened h5file yet, do it now
        if not hasattr(self, 'h5file'):
            self._init_h5file()

        # Randomly select a demonstration
        demo_idx = np.random.randint(0, self.datasetlen)
        demo_key = self.demo_keys[demo_idx]
        demo_data = self.h5file['data'][demo_key]
        
        # Get number of frames in this demo
        num_frames = demo_data.attrs['num_samples']

        action_index = np.random.randint(0, num_frames)

        imgs = demo_data['obs/frontview_image']
        states_dataset = demo_data['obs/state']
        actions_dataset = demo_data['action']
        actions = actions_dataset[action_index]

        action_2d_pixel_x_indices = get_action_2d_pixel_x_indices()
        action_2d_pixel_y_indices = get_action_2d_pixel_y_indices()

        # Get the visibility mask for the left and right hands
        visibility_mask_left = torch.ones(1)
        visibility_mask_right = torch.ones(1)
        x_actions = actions[action_2d_pixel_x_indices]
        y_actions = actions[action_2d_pixel_y_indices]
        if x_actions.min() <= 0 or y_actions.min() <= 0 or self.actions_mask_left[demo_key][action_index] == 0:
            visibility_mask_left[0] = 0
        if x_actions.max() >= MAX_X_PIXEL_VALUE or y_actions.max() >= MAX_Y_PIXEL_VALUE or self.actions_mask_right[demo_key][action_index] == 0:
            visibility_mask_right[0] = 0

        actions = (actions - self.actions_mean) / self.actions_std

        lang_embedding = demo_data['obs/language_embedding'][0]
        contact_gmm_left = demo_data['contact/gmm_contacts_left'][:][action_index]
        sorted_indices_left = np.argsort(contact_gmm_left[:, 0])
        contact_gmm_left = contact_gmm_left[sorted_indices_left]
        contact_gmm_right = demo_data['contact/gmm_contacts_right'][:][action_index]
        sorted_indices_right = np.argsort(contact_gmm_right[:, 0])
        contact_gmm_right = contact_gmm_right[sorted_indices_right]
        intersected_bbox_left = demo_data['contact/intersected_bbox_left'][:][action_index]
        intersected_bbox_right = demo_data['contact/intersected_bbox_right'][:][action_index]
        contact_mask_left = torch.ones(1)
        contact_mask_right = torch.ones(1)
        object_mask_left = torch.ones(1)
        object_mask_right = torch.ones(1)
        if np.any(contact_gmm_left) == 0 or self.contact_left_mask[demo_key][action_index] == 0:
            contact_mask_left = torch.zeros(1)
        if np.any(contact_gmm_right) == 0 or self.contact_right_mask[demo_key][action_index] == 0:
            contact_mask_right = torch.zeros(1)
        if np.any(intersected_bbox_left) == 0 or self.obj_left_mask[demo_key][action_index] == 0:
            object_mask_left = torch.zeros(1)
        if np.any(intersected_bbox_right) == 0 or self.obj_right_mask[demo_key][action_index] == 0:
            object_mask_right = torch.zeros(1)

        contact_gmm_left = (contact_gmm_left - self.contact_gmm_left_mean) / self.contact_gmm_left_std
        contact_gmm_right = (contact_gmm_right - self.contact_gmm_right_mean) / self.contact_gmm_right_std
        intersected_bbox_left = (intersected_bbox_left - self.obj_info_left_mean) / self.obj_info_left_std
        intersected_bbox_right = (intersected_bbox_right - self.obj_info_right_mean) / self.obj_info_right_std
    
        contact_info_left = torch.cat([contact_mask_left, torch.tensor(contact_gmm_left.reshape(-1), dtype=torch.float32)])
        contact_info_right = torch.cat([contact_mask_right, torch.tensor(contact_gmm_right.reshape(-1), dtype=torch.float32)])
        obj_info_left = torch.cat([object_mask_left, torch.tensor(intersected_bbox_left.reshape(-1), dtype=torch.float32)])
        obj_info_right = torch.cat([object_mask_right, torch.tensor(intersected_bbox_right.reshape(-1), dtype=torch.float32)])

        # Now safe image access
        im = imgs[action_index]
        if self.center_crop:
            extra = im.shape[1] - im.shape[0]
            im = im[:, extra//2:-extra//2, :]

        im = (self.aug(torch.tensor(im.transpose(2, 0, 1), dtype=torch.float32) / 255.0) * 255.0)

        action_2d_rot_grip_indices = get_action_2d_rot_grip_indices()
        action_2d_indices = get_action_2d_pixel_indices()

        if self.include_rot_grip:
            actions = actions[action_2d_rot_grip_indices]
        else:
            actions = actions[action_2d_indices]

        if self.center_crop:
            raise NotImplementedError("Center crop not implemented - need to account for new normalization!!!")
            # # Any action < 100 should now be 0 
            # # Any action > 356 should now be 256
            # # Any action > 100 and < 356 should be action-100
            # actions[action_2d_indices_x] = torch.clamp(actions[action_2d_indices_x], 0, 356)
            # actions[action_2d_indices_x] = actions[action_2d_indices_x] - 100
            # actions[action_2d_indices_x] = torch.clamp(actions[action_2d_indices_x], 0, 256)
            # max_pixel_val = 256.0
        
        state = torch.tensor(states_dataset[action_index], dtype=torch.float32).clone()

        # Return
        return (im, torch.tensor(lang_embedding, dtype=torch.float32).clone(),
                state, actions, visibility_mask_left, visibility_mask_right, contact_info_left, contact_info_right, obj_info_left, obj_info_right)

    def __iter__(self):
        self._init_h5file()
        while True:
            for _ in range(1000):
                yield self._sample()

    def __del__(self):
        if hasattr(self, 'h5file'):
            self.h5file.close()