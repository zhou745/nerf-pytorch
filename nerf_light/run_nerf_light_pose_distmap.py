import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from transformer import vit_toy_patch16_256

import matplotlib.pyplot as plt

from run_nerf_in_the_wild_helpers import *

from dataset import Nerf_blender_light_dataset,Nerf_real_light_dataset, Nerf_llff_dataset
from model import Nerf_density, Nerf_color, Nerf_pose
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def batchify(fn, num_sample, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs):
        result_list = [fn({key: inputs[key][i:i + chunk] for key in inputs.keys()}) for i in
                       range(0, num_sample, chunk)]
        result = {key: torch.cat([result_list[i][key] for i in range(len(result_list))], 0) for key in
                  result_list[0].keys()}
        return (result)

    return ret


def run_network(xyz, dir, light_cond,
                density_fn, color_fn,
                embed_fn_xyz, embed_fn_dir, netchunk=1024 * 64):
    """Prepares inputs and applies network 'fn'.
    xyz: N * 3
    dir: N * 3
    light_cond: N * 200
    """

    encode_xyz = embed_fn_xyz(xyz)
    encode_dir = embed_fn_dir(dir)

    # querry network based on chunk
    output_density_dict = batchify(density_fn, encode_xyz.shape[0], netchunk)({'encode_xyz': encode_xyz})
    color_encode, density = output_density_dict['color_encode'], output_density_dict['density']

    output_color_dict = batchify(color_fn, color_encode.shape[0], netchunk)({'encode_color': color_encode,
                                                                             'encode_dir': encode_dir,
                                                                             'encode_light': light_cond})

    return {'color': output_color_dict['color'],
            'density': density}


def batchify_rays(rays_o, rays_d, viewdirs, light_cond, chunk=1024 * 32, eval_model=False, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    N, num_sample, _ = rays_o.shape

    light_cond_repeat = light_cond.unsqueeze(1).repeat(1, num_sample, 1)
    # falt the rays_o and rays_d
    rays_o_flat = rays_o.flatten(0, 1)
    rays_d_flat = rays_d.flatten(0, 1)
    viewdirs_flat = viewdirs.flatten(0, 1)
    light_cond_flat = light_cond_repeat.flatten(0, 1)

    for i in range(0, N * num_sample, chunk):
        ret = render_rays(rays_o_flat[i:i + chunk], rays_d_flat[i:i + chunk],
                          viewdirs_flat[i:i + chunk], light_cond_flat[i:i + chunk], eval_model=eval_model, **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, chunk=1024 * 32, rays_o=None, rays_d=None, viewdirs=None, light_cond=None,ref_img =None, device=None,
           near=2.0, far=6.0, eval_model=False,gt_light_rate=0.2, **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays_o: array of shape [N, sample, 3]. Ray origin for each example in batch.
      rays_d: array of shape [N, sample, 3]. Ray direction for each example in batch.
      kwargs: network and other training parameters
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """

    rays_o_cuda = rays_o.to(device)
    rays_d_cuda = rays_d.to(device)
    viewdirs_cuda = viewdirs.to(device)

    light_cond_cuda = light_cond.to(device)
    # Render and reshape
    sh = rays_d.shape
    all_ret = batchify_rays(rays_o_cuda, rays_d_cuda, viewdirs_cuda, light_cond_cuda, chunk, eval_model=eval_model,
                            **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, light_cond, hwf, K, chunk, render_kwargs, img_idx=0, gt_imgs=None,ref_img=None, savedir=None,
                gt_light_rate = -0.1, render_factor=0):
    H, W, focal = hwf
    K_use = K
    if render_factor != 0:
        # Render downsampled for speed
        H = int(H * render_factor)
        W = int(W * render_factor)
        if len(focal)==1:
            focal[0] = focal[0] * render_factor
            K_use = np.array([[focal[0], 0, 0.5 * W],
                              [0, focal[0], 0.5 * H],
                              [0, 0, 1]])
        else:
            focal[0] = focal[0] * render_factor
            focal[1] = focal[1] * render_factor
            focal[2] = focal[2] * render_factor
            focal[3] = focal[3] * render_factor
            K_use = np.array([[focal[0], 0, focal[2]],
                              [0, focal[1], focal[3]],
                              [0, 0, 1]])

    # render the rays
    rays_o, rays_d = get_rays(H, W, K_use, render_poses[0, :3, :4])

    # add batch dim to the front
    rays_o = rays_o.unsqueeze(0)
    rays_d = rays_d.unsqueeze(0)
    # flat H W
    rays_o_flat = rays_o.flatten(1, 2)
    rays_d_flat = rays_d.flatten(1, 2)
    viewdirs = rays_d_flat / rays_d_flat.norm(dim=-1, keepdim=True)
    ref_img_run = ref_img[...,:3]
    rgb, disp, acc, extras = render(H, W, K_use, chunk=chunk, rays_o=rays_o_flat, rays_d=rays_d_flat,
                                    viewdirs=viewdirs, light_cond=light_cond, ref_img= ref_img_run, device=light_cond.device,
                                    eval_model=True,gt_light_rate=gt_light_rate, **render_kwargs)

    rgb_2d = rgb.reshape(H, W, -1)
    disp_2d = disp.reshape(H, W, -1)
    print(rgb_2d.shape, disp_2d.shape)

    if savedir is not None:
        rgb_cpu = rgb_2d.cpu().numpy()
        disp_cpu = disp_2d.cpu().numpy()
        rgb8 = to8b(rgb_cpu)
        disp8 = to8b(disp_cpu)
        filename = os.path.join(savedir, '{:03d}.png'.format(img_idx))
        filename_dsip = os.path.join(savedir, '{:03d}_disp.png'.format(img_idx))
        imageio.imwrite(filename, rgb8)
        imageio.imwrite(filename_dsip, disp8)

        if gt_imgs is not None:
            gt_cpu = gt_imgs[0].cpu().numpy()
            gt8 = to8b(gt_cpu)
            filename_gt = os.path.join(savedir, '{:03d}_gt.png'.format(img_idx))
            imageio.imwrite(filename_gt, gt8)

            ref_cpu = ref_img[0].cpu().numpy()
            ref8 = to8b(ref_cpu)
            filename_ref = os.path.join(savedir, '{:03d}_ref.png'.format(img_idx))
            imageio.imwrite(filename_ref, ref8)

    return rgb, disp


def create_nerf(args,train_dataset = None):
    """Instantiate NeRF's MLP model.
    """

    # embed function for xyz
    embed_fn_xyz, input_ch_xyz = get_embedder(args.multires, args.i_embed)

    # embed function for view dir
    embed_fn_dir, input_ch_dir = get_embedder(args.multires_views, args.i_embed)

    # get model for density and encoding
    skips_density = [4]
    skips_color = [2]

    model_pose = Nerf_pose(maximum_pose=args.maximum_pose)
    if args.dataset_type == "real_data":
        model_pose.init_parameter_from_q(data_json_path=os.path.join(args.datadir, 'transforms_train.json'))
    elif train_dataset is not None:
        model_pose.init_parameter_from_dataset(train_dataset)
    else:
        model_pose.init_random_parameter()
    #this model is used to model the camera distortion
    model_dist = Nerf_density(input_ch=input_ch_xyz,
                                 D=args.net_density_depth,
                                 W=args.net_density_width,
                                 skips=skips_density,
                                 output_ch_color=args.color_feature_dim)

    model_density = Nerf_density(input_ch=input_ch_xyz,
                                 D=args.net_density_depth,
                                 W=args.net_density_width,
                                 skips=skips_density,
                                 output_ch_color=args.color_feature_dim)

    model_color = Nerf_color(input_ch_dir=input_ch_dir,
                             input_ch_color=args.color_feature_dim,
                             light_cond=args.light_cond,
                             light_dim=args.light_dim,
                             D=args.net_color_depth,
                             W=args.net_color_width,
                             output_ch=3,
                             skips=skips_color)
    grad_vars = list(model_pose.parameters())
    grad_vars += list(model_density.parameters())
    grad_vars += list(model_color.parameters())

    model_density_fine = None
    model_color_fine = None

    if args.N_importance > 0:
        model_density_fine = Nerf_density(input_ch=input_ch_xyz,
                                          D=args.net_density_depth,
                                          W=args.net_density_width,
                                          skips=skips_density,
                                          output_ch_color=args.color_feature_dim)

        model_color_fine = Nerf_color(input_ch_dir=input_ch_dir,
                                      input_ch_color=args.color_feature_dim,
                                      light_cond=args.light_cond,
                                      light_dim=args.light_dim,
                                      D=args.net_color_depth,
                                      W=args.net_color_width,
                                      output_ch=3,
                                      skips=skips_color)

        grad_vars += list(model_density_fine.parameters())
        grad_vars += list(model_color_fine.parameters())

    # query function for color and density
    network_query_fn = lambda xyz, dir, light_cond,density_fn, color_fn: run_network(xyz, dir, light_cond,
                                                                                      density_fn, color_fn,
                                                                                      embed_fn_xyz=embed_fn_xyz,
                                                                                      embed_fn_dir=embed_fn_dir,
                                                                                      netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model_pose.load_state_dict(ckpt['model_pose'])
        model_density.load_state_dict(ckpt['model_density'])
        model_color.load_state_dict(ckpt['model_color'])

        if model_density_fine is not None:
            model_density_fine.load_state_dict(ckpt['model_density_fine'])
            model_color_fine.load_state_dict(ckpt['model_color_fine'])

    ##########################

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'model_pose': model_pose,
        'model_density': model_density,
        'model_color': model_color,
        'N_samples': args.N_samples,
        'model_density_fine': model_density_fine,
        'model_color_fine': model_color_fine,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw['color'])  # [N_rays, N_samples, 3]
    density = raw['density'].squeeze(-1)
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(density.shape) * raw_noise_std

    alpha = raw2alpha(density + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(rays_o,
                rays_d,
                viewdirs,
                light_cond,
                network_query_fn,
                perturb,
                N_importance,
                model_pose,
                model_density,
                model_color,
                N_samples,
                model_density_fine,
                model_color_fine,
                near=2.0,
                far=6.0,
                white_bkgd=False,
                raw_noise_std=0.,
                eval_model=False):
    """Volumetric rendering.
    Args:
      ray_o_batch: array of shape [batch_size, ...]. origin information necessary
        for sampling along a ray, including: ray origin

      ray_d_batch: array of shape [batch_size, ...]. direction information necessary
        for sampling along a ray, including: ray direction
      view_dir： from which direction the camera is sampling
      network_query_fn: function used for passing queries to network_fn.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      model_density model_color: function. Model for predicting RGB and density at each point
        in space.
      N_samples: int. Number of different times to sample along each ray.
      model_density_fine model_color_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = rays_o.shape[0]
    near_box, far_box = near * torch.ones_like(rays_o[..., :1]), far * torch.ones_like(rays_o[..., :1])

    t_vals = torch.linspace(0., 1., steps=N_samples)
    # get sampling point
    z_vals = near_box * (1. - t_vals) + far_box * (t_vals)

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]
    dirs = viewdirs[:, None, :].repeat(1, N_samples, 1)
    lcond = light_cond[:, None, :].repeat(1, N_samples, 1)
    #     raw = run_network(pts)

    raw = network_query_fn(pts, dirs, lcond, model_density, model_color)
    if eval_model:
        for k in raw.keys():
            raw[k] = raw[k].detach()
        torch.cuda.empty_cache()

    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd)

    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.))
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :,
                                                            None]  # [N_rays, N_samples + N_importance, 3]

        dirs_fine = viewdirs[:, None, :].repeat(1, N_samples + N_importance, 1)
        lcond_fine = light_cond[:, None, :].repeat(1, N_samples + N_importance, 1)
        #         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, dirs_fine, lcond_fine, model_density_fine, model_color_fine)
        if eval_model:
            for k in raw.keys():
                raw[k] = raw[k].detach()
            torch.cuda.empty_cache()
        # raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd)

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}

    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser(default_conf="configs/lego.txt"):
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, default=default_conf,
                        help='config file path')
    parser.add_argument('--num_epoch', type=int, default=400,
                        help='training time')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')

    # training options
    parser.add_argument("--net_density_depth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--net_density_width", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--color_feature_dim", type=int, default=256)
    parser.add_argument("--light_cond", type=int, default=200)
    parser.add_argument("--light_dim", type=int, default=63)

    parser.add_argument("--net_color_depth", type=int, default=4,
                        help='layers in network')
    parser.add_argument("--net_color_width", type=int, default=256,
                        help='channels per layer')

    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024 * 8,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024 * 16,
                        help='number of pts sent through network in parallel, decrease if running out of memory')

    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_epoch", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # dataset options
    parser.add_argument("--maximum_pose", type=int, default=1000,
                        help='max num of pose to store')
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--near", type=float, default=2,
                        help='near point to camera')
    parser.add_argument("--far", type=float, default=6,
                        help='far point to camera')

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')
    parser.add_argument("--quat_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--print_freq_step", type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--save_weights_freq_epoch", type=int, default=1,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--save_testset_freq_epoch", type=int, default=5,
                        help='frequency of testset saving')

    parser.add_argument("--gt_light_rate", type=float, default=0.5,
                        help='frequency of using gt light')

    parser.add_argument("--render_debug", action='store_true')
    parser.add_argument("--train_image_list", type=str,default="")
    parser.add_argument("--val_image_list", type=str, default="")
    parser.add_argument("--test_image_list", type=str, default="")
    parser.add_argument("--scale_pose", type=float,default=1.0)
    return parser


def train():
    # default_conf = "configs/fern.txt"
    # default_conf = "configs/kubric_shoe.txt"
    # default_conf = "configs/light_cond_shoes.txt"
    # default_conf = "configs/single_shoes.txt"
    # default_conf = "configs/env_0_front_pose.txt"
    # default_conf = "configs/env_0_front_dist.txt"
    # default_conf = "configs/fern.txt"
    default_conf = "configs/tree.txt"

    parser = config_parser(default_conf=default_conf)

    args = parser.parse_args()
    num_epoch = args.num_epoch
    print(f"training for {args.num_epoch} epochs")
    # Load data
    K = None
    # currently only synthetic data is considered
    # kubric synthetic：blender_light
    # create dataset
    if args.dataset_type == 'blender_light':
        # image name list is loaded, instead of the image itself
        dataset_train = Nerf_blender_light_dataset(args.datadir,
                                                   args.half_res,
                                                   split='train',
                                                   light_cond_dim=args.light_cond)

        dataset_val = Nerf_blender_light_dataset(args.datadir,
                                                 args.half_res,
                                                 split='val',
                                                 light_cond_dim=args.light_cond)

        dataset_test = Nerf_blender_light_dataset(args.datadir,
                                                  args.half_res,
                                                  split='test',
                                                  light_cond_dim=args.light_cond)

        print('Loaded blender dataset')

        hwf = dataset_train.get_hwf()
        near = args.near
        far = args.far
        print('NEAR FAR', near, far)
    elif  args.dataset_type == 'real_data':
        # image name list is loaded, instead of the image itself
        dataset_train = Nerf_real_light_dataset(args.datadir,
                                                   args.half_res,
                                                   args.quat_res,
                                                   split='train',
                                                   light_cond_dim=args.light_cond,
                                                   image_list=args.train_image_list)

        dataset_val = Nerf_real_light_dataset(args.datadir,
                                                 args.half_res,
                                                 args.quat_res,
                                                 split='val',
                                                 light_cond_dim=args.light_cond,
                                                 image_list=args.val_image_list)

        dataset_test = Nerf_real_light_dataset(args.datadir,
                                                  args.half_res,
                                                  args.quat_res,
                                                  split='test',
                                                  light_cond_dim=args.light_cond,
                                                  image_list=args.test_image_list)

        print('Loaded nerf real dataset')

        hwf = dataset_train.get_hwf()
        near = args.near
        far = args.far
        print('NEAR FAR', near, far)

    elif args.dataset_type == "llff_data":
        dataset_train = Nerf_llff_dataset(args.datadir,
                                          factor = args.factor,
                                          light_cond_dim = args.light_cond)

        dataset_val = Nerf_llff_dataset(args.datadir,
                                          factor = args.factor,
                                          light_cond_dim = args.light_cond)

        dataset_test = Nerf_llff_dataset(args.datadir,
                                          factor = args.factor,
                                          light_cond_dim = args.light_cond)

        hwf = dataset_train.get_hwf()
        print('DEFINING BOUNDS')

        bds = dataset_train.get_bds()
        near = np.ndarray.min(bds) * .9
        far = np.ndarray.max(bds) * 1.

        print('NEAR FAR', near, far)
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W = hwf[0:2]
    focal = hwf[2:]
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None and len(focal) ==1:
        K = np.array([
            [focal[0], 0, 0.5 * W],
            [0, focal[0], 0.5 * H],
            [0, 0, 1]
        ])
    elif K is None and len(focal) ==4:
        K = np.array([
            [focal[0], 0, focal[2]],
            [0, focal[1], focal[3]],
            [0, 0, 1]
        ])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # create the pytorch dataloader for loading images

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args,train_dataset=dataset_train)
    global_step = start
    epoch_step = start // (len(dataset_train) * args.batch_size)

    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # creqate the data loader
    sampler_train = torch.utils.data.RandomSampler(dataset_train, generator=torch.Generator(device="cuda:0"))
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
        prefetch_factor=2,
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, shuffle=False,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
        prefetch_factor=2,
    )

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand

    print('Begin')
    print('TRAIN views are', len(dataset_train))
    print('TEST views are', len(dataset_test))
    print('VAL views are', len(dataset_val))

    # debug use
    if args.render_debug:
        save_path = "./render/env_0_pose/epoch_323_train"
        # save_path = "./render/single_shoes/epoch_6000_test"
        render_dataset(save_path, hwf, K, args, dataset_train, render_kwargs_test, device,
                       offset_idx=0,step_idx=1, num_render=20, light_cond_ratio=None,gt_light_rate=-0.1)
        return

    # Summary writers
    writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    epoch_start = epoch_step
    for i in range(epoch_start, num_epoch):
        # time0 = time.time()
        for data_batch in tqdm(dataloader_train):

            # break #debug use
            images = data_batch['images']
            poses_ori = data_batch['poses']
            light_cond = data_batch['light_cond']
            ref_img = data_batch['ref_img']
            image_idx = data_batch['image_idx']

            if args.white_bkgd:
                images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
                ref_img = ref_img[..., :3] * ref_img[..., -1:] + (1. - ref_img[..., -1:])
            else:
                images = images[..., :3]
                ref_img = ref_img[..., :3]

            #original pose
            N, _, _ = poses_ori.shape
            poses = render_kwargs_train['model_pose'](image_idx)

            # sample rays for each image
            rays_o, rays_d = get_rays_batch(H, W, K, poses)
            # generator sampling coords
            if i < args.precrop_epoch:
                dH = int(H // 2 * args.precrop_frac)
                dW = int(W // 2 * args.precrop_frac)
                coords = torch.stack(
                    torch.meshgrid(
                        torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                        torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW)
                    ), -1)
                if global_step == start:
                    print(
                        f"[Config] Center cropping of size {2 * dH} x {2 * dW} is enabled until epoch {args.precrop_epoch}")
            else:
                coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)),
                                     -1)  # (H, W, 2)

            coords = torch.reshape(coords, [1, -1, 2])  # (H * W, 2)
            coords = coords.repeat(N, 1, 1)
            select_inds = torch.tensor(np.random.choice(coords.shape[1], size=[N, N_rand], replace=False))
            select_inds = select_inds.unsqueeze(-1).repeat(1, 1, 2)
            # select_coords = coords[select_inds].long()  # (N_rand, 2)
            select_coords = torch.gather(coords, 1, select_inds).long()
            rays_o = torch.stack([rays_o[i, select_coords[i, :, 0], select_coords[i, :, 1]] for i in range(N)],
                                 dim=0)  # (N_rand, 3)
            rays_d = torch.stack([rays_d[i, select_coords[i, :, 0], select_coords[i, :, 1]] for i in range(N)],
                                 dim=0)  # (N_rand, 3)
            viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
            target_s = torch.stack([images[i, select_coords[i, :, 0], select_coords[i, :, 1]] for i in range(N)],
                                   dim=0)  # (N_rand, 3)

            #####  Core optimization loop  #####
            rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays_o=rays_o, rays_d=rays_d,
                                            viewdirs=viewdirs, light_cond=light_cond,ref_img=ref_img, device=images.device,
                                            gt_light_rate = args.gt_light_rate, **render_kwargs_train)

            optimizer.zero_grad()
            img_loss = img2mse(rgb, target_s)
            loss = img_loss
            psnr = mse2psnr(img_loss)

            if 'rgb0' in extras:
                img_loss0 = img2mse(extras['rgb0'], target_s)
                loss = loss + img_loss0
                psnr0 = mse2psnr(img_loss0)

            loss.backward()
            optimizer.step()

            # NOTE: IMPORTANT!
            ###   update learning rate   ###
            decay_rate = 0.1
            decay_steps = args.lrate_decay * 1000
            new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate
            ################################

            # dt = time.time() - time0
            global_step += 1
            # write tensorboard
            writer.add_scalar('Loss', loss.item(), global_step)
            writer.add_scalar('PSNR', psnr.item(), global_step)
            writer.add_scalar('Loss_fine', img_loss.item(), global_step)
            writer.add_scalar('Loss_coarse', img_loss0.item(), global_step)

            if global_step % args.print_freq_step == 0:
                tqdm.write(
                    f"[TRAIN] Iter: {global_step} lr: {new_lrate} Loss: {loss.item()}  PSNR: {psnr.item()}, Loss_fine: {img_loss.item()}, Loss_coarse: {img_loss0.item()}")
            # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
            #####           end            #####

        # Rest is logging
        epoch_step += 1
        if epoch_step % args.save_weights_freq_epoch == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'model_density': render_kwargs_train['model_density'].state_dict(),
                'model_color': render_kwargs_train['model_color'].state_dict(),
                'model_pose': render_kwargs_train['model_pose'].state_dict(),
                'model_density_fine': render_kwargs_train['model_density_fine'].state_dict(),
                'model_color_fine': render_kwargs_train['model_color_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        # providing a fake light_cond and rendering
        if epoch_step % args.save_testset_freq_epoch == 0 and epoch_step > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test cases num ', len(dataset_test))
            with torch.no_grad():
                img_idx = 0
                for data_batch in tqdm(dataloader_test):
                    images = data_batch['images'].to(device)
                    image_idx = data_batch['image_idx'].to(device)
                    poses = data_batch['poses'].to(device)
                    poses_model = render_kwargs_test['model_pose'](image_idx).detach()
                    light_cond = data_batch['light_cond'].to(device)
                    ref_img = data_batch['ref_img'].to(device)

                    render_path(poses, light_cond, hwf, K, args.chunk, render_kwargs_test, img_idx=img_idx*2,
                                gt_imgs=images,ref_img =ref_img, savedir=testsavedir, render_factor=1.0)
                    render_path(poses_model, light_cond, hwf, K, args.chunk, render_kwargs_test, img_idx=img_idx*2+1,
                                gt_imgs=images,ref_img =ref_img, savedir=testsavedir, render_factor=1.0)
                    img_idx += 1
            print('Saved test set')
    writer.close()


def render_dataset(save_dir, hwf, K, args, dataset, render_kwargs_test, device, offset_idx=0, step_idx = 1,
                   num_render=10, render_factor=1.0, gt_light_rate = 1.1,light_cond_ratio=None):
    testsavedir = save_dir
    os.makedirs(testsavedir, exist_ok=True)
    print('test cases num ', num_render)
    pose_colmap_list = []
    pose_training_list = []

    with torch.no_grad():
        for img_idx in tqdm(range(num_render)):
            data_batch = dataset[img_idx*step_idx+offset_idx]
            images = data_batch['images'].to(device).unsqueeze(0)
            image_idx = data_batch['image_idx'].to(device).unsqueeze(0)
            poses_colmap = data_batch['poses'].to(device).unsqueeze(0)
            poses = render_kwargs_test['model_pose'](image_idx).detach()
            light_cond = data_batch['light_cond'].to(device).unsqueeze(0)
            ref_img = data_batch['ref_img'].to(device).unsqueeze(0)
            pose_colmap_list.append(poses_colmap[:,:,:-1].detach().cpu())
            pose_training_list.append(poses[:,:-1,:].detach().cpu())

            light_idx = np.where(light_cond[0].cpu().numpy()>0.5)
            print(light_idx,flush=True)
            if light_cond_ratio is not None:
                light_cond[0,light_idx[0][0]] = light_cond_ratio[0]
                light_cond[0,light_idx[0][1]] = light_cond_ratio[1]

            # render_path(poses, light_cond, hwf, K, args.chunk, render_kwargs_test, img_idx=img_idx,
            #             gt_imgs=images, ref_img = ref_img,savedir=testsavedir, render_factor=render_factor,
            #             gt_light_rate= gt_light_rate)
    print('Saved test set')


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')  # using gpu by default
    train()
