import torch
import torch.nn as nn
from torch.nn import init
import math
import copy
import numpy as np
from skimage import measure


class QNet(nn.Module):
    def __init__(self, depth=32):
        super(QNet, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, input_tensor):
        q_feature = self.conv0(input_tensor)
        return q_feature


class KVNet(nn.Module):
    def __init__(self, depth=32):
        super(KVNet, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, input_tensor):
        k_v_feature = self.conv0(input_tensor)
        return k_v_feature


class FeedForwardNet(nn.Module):
    def __init__(self, depth=128):
        super(FeedForwardNet, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, input_tensor):
        out = self.conv0(input_tensor)
        return out


class AttentionNet(nn.Module):
    def __init__(self, depth=64):
        super(AttentionNet, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=depth, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, input_tensor):
        out = self.conv0(input_tensor)
        return out


def _get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for i in range(n)])


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.xavier_normal(m.weight.data)

    if classname.find('ConvTranspose2d') != -1:
        init.xavier_normal(m.weight.data)


def cal_psnr(img1, img2):
    img1_np = np.array(img1)
    img2_np = np.array(img2)
    return measure.compare_psnr(img1_np, img2_np)


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2*(i // 2))/ np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model, h=128, w=226):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    sines = np.sin(angle_rads[:, 0::2])

    cones = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.concatenate([sines, cones], axis=-1).astype(np.float32)
    pos_embedding = torch.from_numpy(0.5*pos_encoding)
    pos = pos_embedding.unsqueeze(2).repeat(1, 1, h * w).reshape(position, d_model, h, w).cuda()
    return pos


class PositionalEmbeddingLearned(nn.Module):
    def __init__(self, embedding_depth=128):
        super(PositionalEmbeddingLearned, self).__init__()
        self.depth = embedding_depth
        self.positional_embedding = nn.Embedding(10, self.depth).cuda()

    def forward(self, shape):
        b, c, h, w = shape
        index = torch.arange(b).cuda()#to('cuda:0')
        position = self.positional_embedding(index)  # 5 * 64
        position = position.unsqueeze(2).repeat(1, 1, h * w).reshape(b, self.depth, h, w)
        return position

def get_model_name(cfg):
    if cfg.w_res:
        s_res = 'w_res-'
    else:
        s_res = 'wo_res-'
    if cfg.w_pos:
        s_pos = 'w_pos-'
        s_pos_kind = cfg.pos_kind
    else:
        s_pos = 'wo_pos-'
        s_pos_kind = 'none'
    s_num_heads = f'{cfg.n_heads}heads-'
    s_num_layers = f'{cfg.n_layers}layers-'
    s_num_dec_frames = f'dec_{cfg.dec_frames}-'
    s_model_type = '-inter' if cfg.model_type == 0 else '-extra'
    model_kind = s_num_heads + s_num_layers + s_num_dec_frames + s_res + s_pos + s_pos_kind + s_model_type
    return model_kind

if __name__ == '__main__':
    x = positional_encoding(3, 64)
    print('debug')