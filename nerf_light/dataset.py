from torch.utils.data import Dataset
import json
import os
import numpy as np
import imageio
import torch
import cv2

class Nerf_blender_light_dataset(Dataset):
    def __init__(self, basedir, half_res=False, split = 'Train',light_cond_dim=200):
        super(Nerf_blender_light_dataset, self).__init__()

        self.split = split
        self.light_cond_dim = light_cond_dim
        with open(os.path.join(basedir, 'transforms_{}.json'.format(self.split)), 'r') as fp:
            meta = json.load(fp)


        self.imgs_name = []
        self.light_cond = []
        self.poses = []

        for frame in meta['frames']:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            self.imgs_name.append(fname)
            self.poses.append(np.array(frame['transform_matrix']).astype(np.float32))
            try:
                self.light_cond.append(np.array(frame['light_cond']).astype(np.long))
            except:
                self.light_cond.append(np.array(0).astype(np.long))

        imgs_tmp = np.array(imageio.imread(fname))
        H, W = imgs_tmp.shape[:2]

        self.H = H
        self.W = W
        self.camera_angle_x = float(meta['camera_angle_x'])
        self.focal = .5 * W / np.tan(.5 * self.camera_angle_x)
        self.half_res = half_res

        if half_res:
            self.H = self.H // 2
            self.W = self.W // 2
            self.focal = self.focal / 2.

    def __len__(self):
        return(len(self.imgs_name))

    def __getitem__(self, idx):

        img = (np.array(imageio.imread(self.imgs_name[idx]))/ 255.).astype(np.float32)
        #interpolate to size self.H self.W
        if self.half_res:
           img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        light_cond_tensor = torch.zeros((self.light_cond_dim),dtype=torch.float32)
        light_cond_tensor[self.light_cond[idx]] = 1.
        data_dict = {'images': torch.tensor(img),
                     'poses': torch.tensor(self.poses[idx]),
                     'light_cond': light_cond_tensor}
        return(data_dict)

    def get_hwf(self):
        return([self.H,self.W,self.focal])