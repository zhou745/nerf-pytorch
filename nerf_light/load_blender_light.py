import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2

trans_t = lambda t: torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()

rot_phi = lambda phi: torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()

rot_theta = lambda th: torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


def load_blender_light_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs_name = []
    all_poses = []
    all_light_cond = []

    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs_name = []
        light_cond = []
        poses = []
        if s == 'train' or testskip == 0:
            skip = 1
        else:
            skip = testskip

        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs_name.append(fname)
            poses.append(np.array(frame['transform_matrix']))
            light_cond.append(np.array(frame['light_cond']))


        counts.append(counts[-1] +len(imgs_name))

        imgs_name = np.array(imgs_name)
        poses = np.array(poses).astype(np.float32)
        light_cond = np.array(light_cond).astype(np.long)

        all_imgs_name.append(imgs_name)
        all_poses.append(poses)
        all_light_cond.append(light_cond)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    imgs = np.concatenate(all_imgs_name, 0)
    poses = np.concatenate(all_poses, 0)
    light_cond = np.concatenate(all_light_cond, 0)

    imgs_tmp = np.array(imageio.imread(fname))
    H, W = imgs_tmp.shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)

    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.


    return imgs, poses, render_poses, [H, W, focal], i_split, light_cond


