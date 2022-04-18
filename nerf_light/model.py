import torch
import torch.nn as nn
import torch.nn.functional as F

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
                      D = 4,
                      W = 512,
                      output_ch = 3,
                      skips=[2]):

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
