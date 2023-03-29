
import torch
import torch.nn as nn
import copy
from module_utils import *
import torch.nn.functional as F
from matplotlib import pyplot as plt


####################################################################################
#########################  definition for encoder  #################################
####################################################################################

class ConvTransformerEncoder(nn.Module):
    def __init__(self, num_layers=5, model_depth=128, num_heads=4,
                 with_residual=True, with_pos=True, pos_kind='sine'):
        super(ConvTransformerEncoder, self).__init__()
        self.encoderlayer = ConvTransformerEncoderLayer(model_depth, num_heads, with_residual=with_residual,
                                                        with_pos=with_pos)
        self.num_layers = num_layers
        self.depth_perhead = model_depth//num_heads
        self.encoder = self.__get_clones(self.encoderlayer, self.num_layers)
        self.positionnet = PositionalEmbeddingLearned(int(model_depth/num_heads))
        self.pos_kind = pos_kind

    def __get_clones(self, module, n):
        return nn.ModuleList([copy.deepcopy(module) for i in range(n)])

    def forward(self, input_tensor):
        out = input_tensor
        if self.pos_kind == 'sine':
            b, l, c, h, w = input_tensor.shape
            pos = positional_encoding(l, self.depth_perhead, h, w)
        elif self.pos_kind == 'learned':
            pos = self.positionnet(input_tensor.shape[1:])
        for layer in self.encoder:
            out = layer(out, pos)
        return out



class ConvTransformerEncoderLayer(nn.Module): # work as a bridge to handle  multi-head
    def __init__(self, model_depth=128, num_heads=4, with_residual=True, with_pos=True):
        super(ConvTransformerEncoderLayer, self).__init__()
        self.depth = model_depth
        self.depth_perhead = int(model_depth/num_heads)
        self.with_residual = with_residual
        self.attention_heads = self.__get_clones(ConvTransformerEncoderLayerOneHead(self.depth_perhead,
                                                                                    with_pos=with_pos), num_heads)
        self.feedforward = FeedForwardNet(self.depth)
        self.GN1 = nn.GroupNorm(num_groups=4, num_channels=model_depth)

    def __get_clones(self, module, n):
        return nn.ModuleList([copy.deepcopy(module) for i in range(n)])

    def forward(self, input_tensor, pos_encoding):
        heads_out = []
        i = 0
        for head in self.attention_heads:
            heads_out.append(head(input_tensor[:, :, i*self.depth_perhead:(i+1)*self.depth_perhead, :, :], pos_encoding))
            i += 1
        if self.with_residual:
            att_out = torch.cat(heads_out, dim=2) + input_tensor # b,l,c,h,w
            b,l,c,h,w = att_out.shape
            att_out = torch.reshape(att_out, (-1,c,h,w))
            out = self.feedforward(att_out) + att_out
        else:
            att_out = torch.cat(heads_out, dim=2)
            b, l, c, h, w = att_out.shape
            att_out = torch.reshape(att_out, (-1, c, h, w))
            out = self.feedforward(att_out)
        out = self.GN1(out)
        out = torch.reshape(out, (b, l, c, h, w))

        return out


class ConvTransformerEncoderLayerOneHead(nn.Module):
    def __init__(self, head_depth=32, with_pos=True):
        super(ConvTransformerEncoderLayerOneHead, self).__init__()
        self.depth_perhead = head_depth
        self.q_featuremap = QNet(self.depth_perhead)
        self.k_v_featuremap = KVNet(self.depth_perhead)
        self.attentionmap = AttentionNet(self.depth_perhead * 2)
        self.feedforward = FeedForwardNet(self.depth_perhead)
        self.with_pos = with_pos

    def forward(self, input_tensor, pos_encoding):
        batch, length, channel, height, width = input_tensor.shape
        input_tensor = torch.reshape(input_tensor, (batch*length, channel, height, width)) # b*l,c,h,w
        q_feature = self.q_featuremap(input_tensor)
        k_feature = v_feature = self.k_v_featuremap(input_tensor)

        q_feature = torch.reshape(q_feature, (batch, length, channel, height, width)) # b,l,c,h,w
        k_feature = torch.reshape(k_feature, (batch, length, channel, height, width)) # b,l,c,h,w
        v_feature = torch.reshape(v_feature, (batch, length, channel, height, width)) # b,l,c,h,w

        if self.with_pos:
            q_feature = (q_feature + pos_encoding)
            k_feature = (k_feature + pos_encoding)
        else:
            q_feature = q_feature
            k_feature = k_feature

        # convolutional self-attention part
        q_feature = q_feature.unsqueeze(dim=2).repeat(1, 1, length, 1, 1, 1) # b,l,l,c,h,w
        k_feature = k_feature.unsqueeze(dim=1).repeat(1, length, 1, 1, 1, 1) # b,l,l,c,h,w
        v_feature = v_feature.unsqueeze(dim=1).repeat(1, length, 1, 1, 1, 1) # b,l,l,c,h,w

        q_k_concat = torch.cat((q_feature, k_feature), dim=3) # b,l,l,2c,h,w
        dim0, dim1, dim2, dim3, dim4, dim5 = q_k_concat.shape

        q_k_concat = torch.reshape(q_k_concat, (dim0 * dim1 * dim2, dim3, dim4, dim5))
        attention_map = self.attentionmap(q_k_concat)
        attention_map = torch.reshape(attention_map, (dim0, dim1, dim2, 1, dim4, dim5))
        attention_map = nn.Softmax(dim=2)(attention_map) # b,l,l,1,h,w
      

        attentioned_v_Feature = attention_map * v_feature # b,l,l,c,h,w
        attentioned_v_Feature = torch.sum(attentioned_v_Feature, dim=2) # b,l,c,h,w

        return attentioned_v_Feature

