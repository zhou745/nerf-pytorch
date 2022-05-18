import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import itertools
from einops import rearrange, reduce, repeat
eps = 1e-8

class Embedding(nn.Module):
    def __init__(self, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.funcs = [torch.sin, torch.cos]
        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, N_freqs - 1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (N_freqs - 1), N_freqs)

    def forward(self, x):
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq * x)]

        return torch.cat(out, -1)


class Nerf_voxel_embed(nn.Module):
    def __init__(self,voxel_embeddim=24, voxel_freqs=6, max_voxels=400000,
                 pcd_path="", scene_center="0.,0.,0.",voxel_size=0.3,neighbor_marks=3,
                 use_xyz_embed = True, xyz_freqs = 10,scale_factor = 1.):
        super(Nerf_voxel_embed, self).__init__()
        #define voxel embed parameters
        self.voxel_embeddim = voxel_embeddim
        self.voxel_freqs = voxel_freqs
        self.max_voxels = max_voxels
        self.pcd_path = pcd_path
        self.scene_center = [float(num) for num in scene_center.split(",")]
        self.voxel_size_ = voxel_size
        self.neighbor_marks = neighbor_marks
        self.use_xyz_embed = use_xyz_embed
        self.xyz_freqs = xyz_freqs
        self.scale_factor=scale_factor

        #set the embeding function
        self.voxel_embed = Embedding(voxel_freqs)
        if self.use_xyz_embed:
            self.xyz_embed = Embedding(self.xyz_freqs)
        else:
            self.xyz_embed = None

        #set the pcd in nn.embed
        self.embedding_space_ftr = nn.Embedding(max_voxels, self.voxel_embeddim)
        self.set_pointclouds()
        print("embeder finished init")

    def set_pointclouds(self):
        # load pointclouds from file
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(self.pcd_path)

        pcd_xyz = np.asarray(pcd.points)

        scene_center = np.array(self.scene_center)
        scale_factor = self.scale_factor
        pcd_xyz = (pcd_xyz - scene_center) / scale_factor

        # assert pcd_xyz.shape[0] <= self.embedding_space_ftr.num_embeddings
        # if isinstance(pcd_xyz, np.ndarray):
        pcd_xyz = torch.from_numpy(pcd_xyz).float()
        pcd_xyz = pcd_xyz.float().cuda()

        # quantize and normalize voxels
        voxel_size = self.voxel_size_ / scale_factor
        self.register_buffer("voxel_size", torch.scalar_tensor(voxel_size))
        self.register_buffer(
            "bounds", torch.stack([pcd_xyz.min(dim=0)[0], pcd_xyz.max(dim=0)[0]])
        )
        # self.bounds = [pcd_xyz.min(dim=0)[0], pcd_xyz.max(dim=0)[0]]
        self.register_buffer("voxel_offset", -self.bounds[0])
        # self.voxel_offset = -self.bounds[0]
        # print(self.bounds)
        self.register_buffer(
            "voxel_shape",
            torch.tensor(
                [
                    ((self.bounds[1][i] - self.bounds[0][i]) / self.voxel_size)
                        .int()
                        .item()
                    + 3
                    for i in range(3)
                ]
            ).cuda(),
        )
        self.register_buffer(
            "voxel_count", torch.scalar_tensor(self.voxel_shape.prod())
        )

        # mark voxel_occupancy voxels
        self.register_buffer(
            "voxel_occupancy",
            torch.zeros(
                (self.voxel_shape[0], self.voxel_shape[1], self.voxel_shape[2])
            ).bool(),
        )

        # quantize xyz to start from index 0
        pcd_quantize = ((pcd_xyz + self.voxel_offset) / self.voxel_size).round().long()

        # mark center and nearby voxels
        print("Filling the voxel_occupancy...")
        invalid_mask = torch.logical_or(
            ((pcd_quantize < 0).sum(1) > 0),
            ((pcd_quantize >= self.voxel_shape).sum(1) > 0),
        )
        pcd_quantize = pcd_quantize[~invalid_mask]
        self.voxel_occupancy[
            pcd_quantize[:, 0], pcd_quantize[:, 1], pcd_quantize[:, 2]
        ] = True

        # marker neighbors by Conv3d
        MARK_NEIGHBOR = self.neighbor_marks
        conv_neighbor = nn.Conv3d(
            1,
            1,
            kernel_size=MARK_NEIGHBOR,
            bias=False,
            padding=(MARK_NEIGHBOR - 1) // 2,
        )
        conv_neighbor.weight.data[:, :, :] = 1
        conv_neighbor = conv_neighbor.cuda()
        orig_shape = self.voxel_occupancy.shape
        self.voxel_occupancy = (
            conv_neighbor(self.voxel_occupancy[None, None, ...].float().cuda())
                .bool()
                .squeeze()
        )
        # mark border surface
        assert self.voxel_occupancy.shape == orig_shape
        # check voxel occupancy
        # dump_voxel_occupancy_map(self.voxel_occupancy, self.voxel_size, scale_factor, scene_center)
        print(
            "Voxel generated:",
            self.voxel_occupancy.shape,
            "Voxel occupancy ratio:",
            self.voxel_occupancy.sum() / self.voxel_occupancy.numel(),
        )
        print("Voxel used:", self.voxel_occupancy.sum())

        # construct voxel idx map for storing sparse voxel
        self.generate_voxel_idx_map()

        self.instance_ftr_C = 8

    def generate_voxel_idx_map(self):
        # construct voxel idx map for storing sparse voxel
        self.register_buffer(
            "voxel_idx_map",
            -torch.ones((self.voxel_shape[0], self.voxel_shape[1], self.voxel_shape[2]))
            .long()
            .cuda(),
        )
        # self.voxel_idx_map =
        idx_occu = torch.nonzero(self.voxel_occupancy)
        assert self.embedding_space_ftr.num_embeddings >= idx_occu.shape[0]
        self.voxel_idx_map[
            idx_occu[:, 0], idx_occu[:, 1], idx_occu[:, 2]
        ] = torch.arange(idx_occu.shape[0]).cuda()


    def forward(self, xyz):
        N, sample, _ = xyz.shape
        xyz = rearrange(xyz, "n1 n2 c -> (n1 n2) c")
        voxel_ftr, inst_ftr = self.compute_voxel_features_sparse(xyz, True, True)
        if self.use_xyz_embed:
            xyz_ftr = self.xyz_embed(xyz)
            xyz_em = torch.cat([voxel_ftr, xyz_ftr], -1)
        else:
            xyz_em = torch.cat([voxel_ftr, inst_ftr], -1)
        #reshape back
        xyz_em = xyz_em.view(N,sample,-1).contiguous()
        return(xyz_em)

    def compute_voxel_features_sparse(
        self, xyz, trilinear_interpolate, positional_embedding=True
    ):
        """
        get voxel features with sparse indexing and trilinear interpolation
        """
        N, _ = xyz.shape
        xyz_scaled = (xyz + self.voxel_offset) / self.voxel_size
        if trilinear_interpolate:
            xyz_quantize = xyz_scaled.floor().long()
            corners = [0, 1]
            xyz_quantize_all = []
            for c in itertools.product(corners, repeat=3):
                xyz_quantize_all += [xyz_quantize + torch.tensor(c).cuda()]
            xyz_quantize_all = torch.cat(xyz_quantize_all, 0)
            voxel_ftr, invalid_mask = self.get_voxel_feature_sparse_from_quantized(
                xyz_quantize_all
            )
            p = xyz_scaled - xyz_quantize.float()
            u, v, w = p[:, 0], p[:, 1], p[:, 2]
            l_u, l_v, l_w = 1 - u, 1 - v, 1 - w
            weights = [
                (l_u) * (l_v) * (l_w),
                (l_u) * (l_v) * (w),
                (l_u) * (v) * (l_w),
                (l_u) * (v) * (w),
                (u) * (l_v) * (l_w),
                (u) * (l_v) * (w),
                (u) * (v) * (l_w),
                (u) * (v) * (w),
            ]
            weights = torch.cat(weights, 0)
            # print(voxel_ftr.shape, weights.shape)
            voxel_ftr = (
                (voxel_ftr * weights.view(-1, 1)).view(8, N, -1).sum(0, keepdim=False)
            )
            # only when all eight voxels are marked as invalid
            invalid_mask = invalid_mask.view(8, N).int().sum(0, keepdim=False) == 8
            # if self.training:
            #     invalid_mask = ~randomly_set_occupancy_mask_to_true(~invalid_mask)
            # voxel_ftr[invalid_mask] = 0
        else:
            xyz_quantize = xyz_scaled.round().long()
            voxel_ftr, invalid_mask = self.get_voxel_feature_sparse_from_quantized(
                xyz_quantize
            )
            # if self.training:
            #     invalid_mask = ~randomly_set_occupancy_mask_to_true(~invalid_mask)
            # voxel_ftr[invalid_mask] = 0
        _, C = voxel_ftr.shape
        scene_x, instance_x = torch.split(
            voxel_ftr, [C - self.instance_ftr_C, self.instance_ftr_C], dim=-1
        )
        # return self.embedding_final(voxel_ftr)
        if positional_embedding:
            return self.voxel_embed(scene_x), self.voxel_embed(instance_x)
        else:
            return voxel_ftr

    def get_voxel_feature_sparse_from_quantized(self, xyz_quantize):
        """
        get voxel features from quantized xyz coord
        """
        # remove points out of bound
        invalid_mask = torch.logical_or(
            (xyz_quantize < 0).sum(1) > 0, (xyz_quantize >= self.voxel_shape).sum(1) > 0
        )
        xyz_quantize[invalid_mask] = 0

        # get sparse voxel indices from quantized coord and voxel_idx_map
        embedding_idx = self.voxel_idx_map[
            xyz_quantize[:, 0], xyz_quantize[:, 1], xyz_quantize[:, 2]
        ]
        # remove idx==-1, which means empty
        empty_mask = embedding_idx < 0
        invalid_mask = torch.logical_or(invalid_mask, empty_mask)
        # just a placeholder idx
        embedding_idx[invalid_mask] = self.embedding_space_ftr.num_embeddings - 1
        voxel_ftr = self.embedding_space_ftr(embedding_idx)
        voxel_ftr[invalid_mask] = 0
        return voxel_ftr, invalid_mask
