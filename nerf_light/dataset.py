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
        self.ref_imgs = []

        for frame in meta['frames']:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            self.imgs_name.append(fname)
            self.poses.append(np.array(frame['transform_matrix']).astype(np.float32))

            if "light_cond" in frame.keys():
                if isinstance(frame['light_cond'],int):
                    self.light_cond.append(np.array([frame['light_cond']]).astype(np.long))
                elif isinstance(frame['light_cond'],list):
                    self.light_cond.append(np.array(frame['light_cond']).astype(np.long))
                else:
                    raise RuntimeError('unknown light cond type')
            else:
                self.light_cond.append(np.array(-1).astype(np.long))

            if "ref_img" in frame.keys():
                self.ref_imgs.append(np.array(frame['ref_img']).astype(np.long))
            else:
                self.ref_imgs.append([])

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
        for l_idx in self.light_cond[idx]:
            if l_idx>=0:
                light_cond_tensor[l_idx] = 1.

        if len(self.ref_imgs[idx])>0:
            ref_idx = self.ref_imgs[idx][np.random.randint(0,len(self.ref_imgs[idx]))]
            img_ref = (np.array(imageio.imread(self.imgs_name[ref_idx])) / 255.).astype(np.float32)
        else:
            img_ref = np.zeros(img.shape).astype(np.float32)

        data_dict = {'images': torch.tensor(img),
                     'poses': torch.tensor(self.poses[idx]),
                     'light_cond': light_cond_tensor,
                     'ref_img':img_ref}
        return(data_dict)

    def get_hwf(self):
        return([self.H,self.W,self.focal])