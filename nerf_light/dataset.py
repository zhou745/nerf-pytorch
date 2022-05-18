from torch.utils.data import Dataset
import json
import os
import numpy as np
import imageio
import torch
import cv2
from scipy.spatial.transform import Rotation

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
            if self.half_res:
                img_ref = cv2.resize(img_ref, (self.W, self.H), interpolation=cv2.INTER_AREA)
        else:
            img_ref = np.zeros(img.shape).astype(np.float32)

        data_dict = {'images': torch.tensor(img),
                     'poses': torch.tensor(self.poses[idx]),
                     'light_cond': light_cond_tensor,
                     'ref_img': torch.tensor(img_ref)}
        return(data_dict)

    def get_hwf(self):
        return([self.H,self.W,self.focal])


class Nerf_real_light_dataset(Dataset):
    def __init__(self, basedir, split = 'Train',scale_res = 1.,
                 light_cond_dim=200,image_list = "", scale_pose = 3.0,scene_center = "0.,0.,0.",exclude_image_list = ""):
        super(Nerf_real_light_dataset, self).__init__()

        self.split = split
        self.light_cond_dim = light_cond_dim
        with open(os.path.join(basedir, 'transforms_{}.json'.format(self.split)), 'r') as fp:
            meta = json.load(fp)
        self.pose_avg = np.concatenate(
            [np.eye(3), np.array([float(it) for it in scene_center.split(",")])[:, None]], 1
        )

        self.imgs_name = []
        self.light_cond = []
        self.poses = []
        self.ref_imgs = []
        if len(image_list)>0:
            if ":" not in image_list:
                image_list_ = [f"r_{item}.png" for item in image_list.split(" ")]
            else:
                start, end, step = image_list.split(":")
                image_list_ = [f"r_{item}.png" for item in range(int(start), int(end), int(step))]
        else:
            image_list_ = []
        if len(exclude_image_list)>0:
            exclude_list_ = [f"r_{item}.png" for item in exclude_image_list.split(" ")]
        else:
            exclude_list_ = []

        for frame in meta['frames']:
            if len(image_list_)>0 and frame['file_path'] not in image_list_:
                   continue
            #exclude some val images
            if self.split=="train" and frame['file_path'] in exclude_list_:
                continue
            #check pose
            fname = os.path.join(basedir, split+"/"+frame['file_path'])
            #get quatarion
            pose = np.zeros((4,4),dtype = float)
            R = Rotation.from_quat(frame['Q'][1:]+frame['Q'][0:1]).as_matrix().astype(np.float32)
            T = np.array(frame['T'],dtype=np.float32)
            #convert the w2c to c2w

            pose[0:3,0:3] = np.transpose(R)
            pose[0:3,3] = np.matmul(-np.transpose(R),T)
            pose[3,3] = 1.

            fix_rot = np.array([1, 0, 0, 0, -1, 0, 0, 0, -1]).reshape(3, 3)
            pose[:3, :3] = pose[:3, :3] @ fix_rot
            # centralize and rescale
            pose = center_pose_from_avg(self.pose_avg, pose)
            pose[:3, 3] = pose[:3, 3]/scale_pose
            pose = pose.astype(np.float32)
            self.imgs_name.append(fname)
            self.poses.append(pose)

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
        #camera para meter
        f_camera = open(os.path.join(basedir, 'intrinsic.json'))
        camera_para = json.load(f_camera)
        f_camera.close()
        K = np.array(camera_para['K'], dtype=np.float)


        self.scale_res = scale_res

        self.H = self.H // self.scale_res
        self.W = self.W // self.scale_res
        self.fx = K[0,0]/self.scale_res
        self.fy = K[1, 1]/self.scale_res
        self.cx = K[0,2]/self.scale_res
        self.cy = K[1, 2]/self.scale_res
        border = 20//self.scale_res
        self.bmask = np.ones((self.H, self.W))
        self.bmask[:border, :] = 0
        self.bmask[-border:, :] = 0
        self.bmask[:, :border] = 0
        self.bmask[:, -border:] = 0

    def __len__(self):
        return(len(self.imgs_name))

    def __getitem__(self, idx):

        img = (np.array(imageio.imread(self.imgs_name[idx]))/ 255.).astype(np.float32)
        image_idx = int(self.imgs_name[idx].rstrip(".png").split("_")[-1])
        #interpolate to size self.H self.W

        img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        light_cond_tensor = torch.zeros((self.light_cond_dim),dtype=torch.float32)
        for l_idx in self.light_cond[idx]:
            if l_idx>=0:
                light_cond_tensor[l_idx] = 1.
        img_ref = np.zeros(img.shape).astype(np.float32)

        data_dict = {'images': torch.tensor(img),
                     'poses': torch.tensor(self.poses[idx]),
                     'light_cond': light_cond_tensor,
                     'ref_img': torch.tensor(img_ref),
                     'image_idx': torch.tensor(image_idx),
                     'valide_mask':torch.tensor(self.bmask,dtype=torch.bool)}
        return(data_dict)

    def get_hwf(self):
        return([self.H,self.W,self.fx,self.fy,self.cx,self.cy])


def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    from shutil import copy
    from subprocess import check_output

    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir

    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100. / r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue

        print('Minifying', r, basedir)

        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)

        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)

        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def poses_avg(poses):
    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w

def recenter_poses(poses):

    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses

class Nerf_llff_dataset(Dataset):
    def __init__(self,basedir, factor=8, recenter=True, bd_factor=.75, light_cond_dim=20):

        super(Nerf_llff_dataset, self).__init__()

        #get suffix and generate img list
        sfx = ''

        if factor is not None:
            sfx = '_{}'.format(factor)
            _minify(basedir, factors=[factor])
            factor = factor

        imgdir = os.path.join(basedir, 'images' + sfx)
        if not os.path.exists(imgdir):
            print(imgdir, 'does not exist, returning')
            return

        self.imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))
                          if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
        #get poses
        poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
        self.poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
        self.bds = poses_arr[:, -2:].transpose([1, 0])

        if self.poses.shape[-1] != len(self.imgfiles):
            print('Mismatch between imgs {} and poses {} !!!!'.format(len(self.imgfiles), self.poses.shape[-1]))
            return

        sh = imageio.imread(self.imgfiles[0]).shape
        self.poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
        self.poses[2, 4, :] = self.poses[2, 4, :] * 1. / factor
        self.poses = np.concatenate([self.poses[:, 1:2, :], -self.poses[:, 0:1, :], self.poses[:, 2:, :]], 1)
        self.poses = np.moveaxis(self.poses, -1, 0).astype(np.float32)

        self.bds = np.moveaxis(self.bds, -1, 0).astype(np.float32)
        # Rescale if bd_factor is provided
        sc = 1. if bd_factor is None else 1. / (self.bds.min() * bd_factor)
        self.poses[:, :3, 3] *= sc
        self.bds *= sc

        if recenter:
            self.poses = recenter_poses(self.poses)

        self.light_cond_dim = light_cond_dim

        self.H, self.W, self.focal = self.poses[0,:3,-1]

    def get_bds(self):
        return(self.bds)

    def __len__(self):
        return(len(self.imgfiles))

    def __getitem__(self, idx):

        img = (np.array(imageio.imread(self.imgfiles[idx]))/ 255.).astype(np.float32)
        image_idx = idx
        #interpolate to size self.H self.W

        light_cond_tensor = torch.zeros((self.light_cond_dim),dtype=torch.float32)
        light_cond_tensor[0] = 1.

        img_ref = np.zeros(img.shape).astype(np.float32)

        data_dict = {'images': torch.tensor(img),
                     'poses': torch.tensor(self.poses[idx]),
                     'bds': torch.tensor(self.bds[idx]),
                     'light_cond': light_cond_tensor,
                     'ref_img': torch.tensor(img_ref),
                     'image_idx': torch.tensor(image_idx)}
        return(data_dict)

    def get_hwf(self):
        return([self.H,self.W,self.focal])


def center_pose_from_avg(pose_avg, pose):
    pose_avg_homo = np.eye(4)
    pose_avg_homo[
        :3
    ] = pose_avg  # convert to homogeneous coordinate for faster computation
    # by simply adding 0, 0, 0, 1 as the last row
    pose_homo = np.eye(4)
    pose_homo[:3] = pose[:3]
    pose_centered = np.linalg.inv(pose_avg_homo) @ pose_homo  # (4, 4)
    # pose_centered = pose_centered[:, :3] # (N_images, 3, 4)
    return pose_centered

class Nerf_voxel_dataset(Dataset):
    def __init__(self, basedir, scale_res = 2, split = 'Train',
                 light_cond_dim=200,image_list = "", scale_pose = 3.0,scene_center="0.,0.,0."):
        super(Nerf_voxel_dataset, self).__init__()

        self.split = split
        self.light_cond_dim = light_cond_dim
        self.pose_avg = np.concatenate(
            [np.eye(3), np.array([float(it) for it in scene_center.split(",")])[:, None]], 1
        )
        with open(os.path.join(basedir, 'transforms_{}.json'.format(self.split)), 'r') as fp:
            meta = json.load(fp)


        self.imgs_name = []
        self.light_cond = []
        self.poses = []

        if len(image_list)>0:
            if ":" not in image_list:
                image_list_ = ["{:04d}".format(int(item)) for item in image_list.split(" ")]
            else:
                start, end, step = image_list.split(":")
                image_list_ = ["{:04d}".format(item) for item in range(int(start),int(end),int(step))]
        else:
            image_list_ = []

        for frame in meta['frames']:
            if len(image_list_)>0 and frame['file_path'] not in image_list_:
                   continue

            #convert the pose where z goes to the opposite direction
            pose = np.array(frame["transform_matrix"])
            #ignore this sample due to numeric reason
            if np.isnan(pose.sum()) or np.isinf(pose.sum()):
                continue

            fix_rot = np.array([1, 0, 0, 0, -1, 0, 0, 0, -1]).reshape(3, 3)
            pose[:3, :3] = pose[:3, :3] @ fix_rot
            # centralize and rescale
            pose = center_pose_from_avg(self.pose_avg, pose)
            pose[:3, 3] /= scale_pose
            self.poses.append(pose.astype(np.float32))
            fname = os.path.join(basedir, split+"/"+frame['file_path']+".png")
            self.imgs_name.append(fname)
            #get quatarion

            if "light_cond" in frame.keys():
                if isinstance(frame['light_cond'],int):
                    self.light_cond.append(np.array([frame['light_cond']]).astype(np.long))
                elif isinstance(frame['light_cond'],list):
                    self.light_cond.append(np.array(frame['light_cond']).astype(np.long))
                else:
                    raise RuntimeError('unknown light cond type')
            else:
                self.light_cond.append(np.array([0]).astype(np.long))

        imgs_tmp = np.array(imageio.imread(fname))
        H, W = imgs_tmp.shape[:2]
        self.H = H
        self.W = W
        #camera para meter
        K = [0.5 * W / np.tan(0.5 * meta["camera_angle_x"])]

        self.scale_res = scale_res
        self.fx = K[0]/self.scale_res
        self.fy = K[0]/self.scale_res
        self.cx = (W/2)/self.scale_res
        self.cy = (H/2)/self.scale_res
        self.H = H//self.scale_res
        self.W = W//self.scale_res
        border = 20
        self.bmask = np.ones((H, W))
        self.bmask[:border, :] = 0
        self.bmask[-border:, :] = 0
        self.bmask[:, :border] = 0
        self.bmask[:, -border:] = 0

    def __len__(self):
        return(len(self.imgs_name))

    def __getitem__(self, idx):

        img = (np.array(imageio.imread(self.imgs_name[idx]))/ 255.).astype(np.float32)
        image_idx = idx
        #interpolate to size self.H self.W
        img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        light_cond_tensor = torch.zeros((self.light_cond_dim),dtype=torch.float32)
        for l_idx in self.light_cond[idx]:
            if l_idx>=0:
                light_cond_tensor[l_idx] = 1.

        data_dict = {'images': torch.tensor(img),
                     'poses': torch.tensor(self.poses[idx]),
                     'light_cond': light_cond_tensor,
                     'image_idx': torch.tensor(image_idx),
                     'valide_mask': torch.tensor(self.bmask,dtype=torch.bool),
                     'image_name': self.imgs_name[idx]}
        return(data_dict)

    def get_hwf(self):
        return([self.H,self.W,self.fx,self.fy,self.cx,self.cy])



