import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from scipy.spatial.transform import Rotation

eps = 1e-8

class Nerf_density(nn.Module):
    def __init__(self, input_ch = 63,
                       D = 8,
                       W = 256,
                       skips=[4],
                       output_ch_color = 256):

        super(Nerf_density,self).__init__()

        self.D = D
        self.W = W
        self.skips = skips
        self.input_ch = input_ch
        self.output_ch_color = output_ch_color

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                                        range(D - 1)])

        self.encoding_head = nn.Linear(W, self.output_ch_color)
        self.density_head = nn.Linear(W,1)

    def forward(self,input_dict):

        #forward the backbone
        encode_xyz = input_dict['encode_xyz']
        h = encode_xyz
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([encode_xyz, h], -1)

        #compute the encoded color and shape
        color_encode = self.encoding_head(h)

        #compute the density
        density = self.density_head(h)

        return({
            "color_encode": color_encode,
            "density": density
        })

class Nerf_color(nn.Module):
    def __init__(self,input_ch_dir=63,
                      input_ch_color = 256,
                      light_cond = 200,
                      light_dim = 63,
                      D = 8,
                      W = 256,
                      output_ch = 3,
                      skips=[4]):

        super(Nerf_color, self).__init__()

        self.D = D
        self.W = W
        self.input_ch_dir = input_ch_dir
        self.input_ch_color = input_ch_color
        self.light_cond = light_cond
        self.light_dim = light_dim

        self.output_ch = output_ch
        self.skips = skips

        self.input_ch_all = self.input_ch_dir + self.input_ch_color + self.light_dim

        self.encode_light = nn.Linear(self.light_cond,self.light_dim,bias=False)
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch_all, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch_all, W) for i in
                                        range(D - 1)])

        self.color_head = nn.Linear(W, self.output_ch)

    def forward(self, input_dict):

        encode_color, encode_dir, encode_light = input_dict['encode_color'], input_dict['encode_dir'],input_dict['encode_light']
        #encode light first
        light_feature = self.encode_light(encode_light)

        input_feature = torch.cat([encode_color,encode_dir,light_feature],dim=-1)
        # forward the backbone
        h = input_feature
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_feature, h], -1)

        #predict colors
        color = self.color_head(h)

        return({
            'color': color
        })


#attention based
class Nerf_light(nn.Module):
    def __init__(self,D = 4,
                      W = 256,
                      light_cond = 200):

        super(Nerf_light, self).__init__()

        self.D = D
        self.W = W
        self.light_cond = light_cond

        self.conv_encode = nn.Conv2d(in_channels=3,out_channels=W,stride=16,kernel_size=16)

        self.conv_list = nn.ModuleList(
            [nn.Conv2d(in_channels= W, out_channels=W,kernel_size=3,stride=2) for i in range(D - 2)])

        self.color_head = nn.Linear(W, self.light_cond)

    def forward(self, input_dict):

        ref_imgs = input_dict['ref_imgs']
        #encode light first
        img_features = self.conv_encode(ref_imgs)

        for i, l in enumerate(self.pts_linears):
            h = self.conv_list[i](img_features)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([img_features, h], -1)

        #predict colors
        color = self.color_head(h)

        return({
            'color': color
        })

#Skew matrix
class Skew(nn.Module):
    def __init__(self):
        super(Skew,self).__init__()

        self.mapping_matrix = torch.tensor([[[0.,0.,0.],[0.,0.,-1.],[0.,1.,0.]],
                                            [[0.,0.,1],[0.,0.,0.],[-1.,0.,0.]],
                                            [[0.,-1,0.],[1.,0.,0.],[0.,0.,0.]]],dtype = torch.float32)

    def forward(self, X):
        #move mapping_matrix to device
        self.mapping_matrix.to(X.device)

        A = (self.mapping_matrix[None,:,:,:]*X[:,None,None,:]).sum(dim=-1)
        return A

#nerf pose net
class Nerf_pose(nn.Module):
    def __init__(self, maximum_pose = 1000):
        super(Nerf_pose, self).__init__()
        self.maximum_pose = maximum_pose
        #rotation axis
        self.pose_embeding_v = nn.Embedding(maximum_pose,3)
        #rotation angle
        self.pose_embeding_alpha = nn.Embedding(maximum_pose,1)
        #translation vactor
        self.pose_embeding_T = nn.Embedding(maximum_pose,3)

        self.skew = Skew()

    def forward(self, image_idx):
        #create K matrix, later use own cuda kernel
        rotation_v = self.pose_embeding_v(image_idx)
        #normailize to 1
        rotation_v = rotation_v/(rotation_v.norm(dim=-1,keepdim=True)+eps)

        rotation_alpha = self.pose_embeding_alpha(image_idx)
        translation = self.pose_embeding_T(image_idx)

        skew_v = self.skew(rotation_v)

        rotation = torch.eye(3).to(rotation_v.device)[None,:,:]+\
                   skew_v*(torch.sin(rotation_alpha)[:,:,None])+\
                   torch.matmul(skew_v,skew_v)*(1-torch.cos(rotation_alpha)[:,:,None])

        N, _, _ = rotation.shape
        #create K matrix
        K = torch.eye(4).to(rotation_v.device).unsqueeze(0).repeat(N,1,1)
        K[:,0:3,0:3] = rotation
        K[:,0:3,3] = translation
        return(K)

    def init_parameter_from_q(self,data_json_path,is_w2c = True):
        #read the json file and get colmap generated pose
        fp = open(data_json_path)
        meta = json.load(fp)
        fp.close()


        rotation_T = [[0.,0.,0.] for i in range(self.maximum_pose)]

        frames = meta['frames']
        Q = [[1.,0.,0.,0.] for i in range(self.maximum_pose)]
        for frame in frames:
            idx = int(frame['file_path'].rstrip(".png").split("_")[-1])
            if is_w2c:
                Q[idx] = np.array(frame['Q'],dtype=np.float32)
                Q[idx][1:] = -Q[idx][1:]
                #rotation
                tmp_R = Q[idx].tolist()
                R = Rotation.from_quat(tmp_R[1:]+tmp_R[0:1]).as_matrix().astype(dtype=np.float32)
                tmp_v = -np.matmul(R,np.array(frame['T'],dtype=np.float32))
                rotation_T[idx] = tmp_v
            else:
                Q[idx] = np.array(frame['Q'], dtype=np.float32)
                rotation_T[idx] = np.array(frame['T'], dtype=np.float32)

        Q_np = np.stack(Q,axis=0)
        rotation_T = np.stack(rotation_T,axis=0)
        rotation_alpha = 2*np.arccos(Q_np[:,0:1])
        rotation_v = Q_np[:,1:]/(np.linalg.norm(Q_np[:,1:],ord=2,axis=-1,keepdims=True)+eps)

        self.pose_embeding_v.weight.data.copy_(torch.tensor(rotation_v))
        self.pose_embeding_alpha.weight.data.copy_(torch.tensor(rotation_alpha))
        self.pose_embeding_T.weight.data.copy_(torch.tensor(rotation_T))

    def init_parameter_from_dataset(self,dataset):
        number_data = len(dataset)
        rotation_T = [[0., 0., 0.] for i in range(self.maximum_pose)]
        Q = [[1., 0., 0., 0.] for i in range(self.maximum_pose)]

        for idx in range(number_data):
            poses = dataset.poses[idx][0:3,0:4]
            rotation_T[idx] = poses[0:3,3]
            tmp_q = Rotation.from_matrix(poses[0:3,0:3]).as_quat().tolist()
            Q[idx] = np.array(tmp_q[3:4]+tmp_q[0:3],dtype=np.float32)
            # Q[idx] = np.array(tmp_q, dtype=np.float32)

        Q_np = np.stack(Q, axis=0)
        rotation_T = np.stack(rotation_T, axis=0)
        rotation_alpha = 2 * np.arccos(Q_np[:, 0:1])
        rotation_v = Q_np[:, 1:] / (np.linalg.norm(Q_np[:, 1:], ord=2, axis=-1, keepdims=True) + eps)

        self.pose_embeding_v.weight.data.copy_(torch.tensor(rotation_v))
        self.pose_embeding_alpha.weight.data.copy_(torch.tensor(rotation_alpha))
        self.pose_embeding_T.weight.data.copy_(torch.tensor(rotation_T))

    def init_random_parameter(self, scale = 0.1, bias = 0.05):
        shape_v = self.pose_embeding_v.weight.data.shape
        shape_alpha = self.pose_embeding_alpha.weight.data.shape
        shape_T = self.pose_embeding_T.weight.data.shape

        self.pose_embeding_v.weight.data.copy_((torch.rand(shape_v)-bias)*scale)
        self.pose_embeding_alpha.weight.data.copy_((torch.rand(shape_alpha)-bias)*scale)
        self.pose_embeding_T.weight.data.copy_((torch.rand(shape_T)-bias)*scale)