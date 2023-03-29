import torch
import torch.nn as nn
import numpy as np
from module import *

# LSTM
class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class LSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(LSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


# ------------------------------------------------------------------------------------------------

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv= nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv(nn.Module):
    '''
    inconv only changes the number of channels
    '''
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            double_conv(in_ch, out_ch),
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()
        self.bilinear=bilinear
        if self.bilinear:
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(in_ch, in_ch//2, 1),)
        else:
            self.up = nn.ConvTranspose2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.up(x)
        # x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class up_unet(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up_unet, self).__init__()
        self.bilinear=bilinear
        if self.bilinear:
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(in_ch, in_ch//2, 1),)
        else:
            self.up =  nn.ConvTranspose2d(in_channels=in_ch, out_channels=in_ch // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class Spatial_Encoder(nn.Module):
    def  __init__(self, nums_hidden, channel_num):
        super(Spatial_Encoder, self).__init__()

        self.inc = inconv(channel_num, nums_hidden[0])
        self.down1 = down(nums_hidden[0], nums_hidden[1])
        self.down2 = down(nums_hidden[1], nums_hidden[2])
        # self.down3 = down(nums_hidden[2], nums_hidden[3])


    def forward(self, x):
        # print(x.shape)
        x = self.inc(x)
        # print(x.shape)
        x = self.down1(x)
        x = self.down2(x)
        # x = self.down3(x)

        return x

class Spatial_Decoder(nn.Module):
    def __init__(self, nums_hidden, channel_num):
        super(Spatial_Decoder, self).__init__()

        # self.up1 = up(nums_hidden[3], nums_hidden[2])
        self.up2 = up(nums_hidden[2], nums_hidden[1])
        self.up3 = up(nums_hidden[1], nums_hidden[0])

        self.out = outconv(nums_hidden[0], channel_num)

    def forward(self, x):
        # x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.out(x)

        return x

class ConvTransformer_recon_correct(nn.Module):
    def __init__(self, tot_raw_num, nums_hidden, num_layers=1, num_dec_frames=1, num_heads=4, with_residual=True,
                 with_pos=True, pos_kind='sine', mode=0, use_flow=True):
        super(ConvTransformer_recon_correct, self).__init__()
        self.raw_channel_num = 3  # RGB channel no.
        self.of_channel_num = 2


        # self.feature_embedding = FeatureEmbedding(model_depth)
        self.feature_embedding = Spatial_Encoder(nums_hidden, self.raw_channel_num)
        self.encoder = ConvTransformerEncoder(num_layers=num_layers, model_depth=nums_hidden[-1], num_heads=num_heads,
                                              with_residual=with_residual, with_pos=with_pos, pos_kind=pos_kind)

        self.prediction = Spatial_Decoder(nums_hidden, self.raw_channel_num)

        if use_flow:
            self.feature_embedding_of = Spatial_Encoder(nums_hidden, self.raw_channel_num)
            self.encoder_of = ConvTransformerEncoder(num_layers=num_layers, model_depth=nums_hidden[-1],
                                                     num_heads=num_heads,
                                                     with_residual=with_residual, with_pos=with_pos, pos_kind=pos_kind)

            self.prediction_of = Spatial_Decoder(nums_hidden, self.of_channel_num)

        self.task = mode
        self.num_dec_frames = num_dec_frames
        self.tot_raw_num = tot_raw_num
        self.tot_of_num = tot_raw_num
        self.use_flow = use_flow
        self.nums_hidden = nums_hidden


    def forward(self, input, of_targets_full):


        b,c_in,h,w = input.shape
        assert c_in == self.raw_channel_num*self.tot_raw_num

        # convert to 5 dimensions for inputs
        input = input.permute(0, 2, 3, 1).contiguous() # b,h,w,c_in
        new_shape_input = input.size()[:-1] + (self.tot_raw_num, self.raw_channel_num) # b,h,w,c,l
        input = input.view(*new_shape_input)
        input = input.permute(0, 3, 4, 1, 2).contiguous().cuda() # b,l,c,h,w

        of_targets_full = of_targets_full.permute(0, 2, 3, 1).contiguous()
        new_shape_of_targets = of_targets_full.size()[:-1] + (self.tot_of_num, self.of_channel_num)
        of_targets_full = of_targets_full.view(*new_shape_of_targets)
        of_targets_full = of_targets_full.permute(0, 3, 4, 1, 2).contiguous().cuda()


        # interpolation
        input_frames = input
        raw_targets = input  # [...,1:]
        input_frames = torch.reshape(input_frames, (-1, self.raw_channel_num, h, w))

        img_tensor = self.feature_embedding(input_frames) # b*l,c_f,h,w
        _, c_f, h_small, w_small = img_tensor.shape
        img_tensor = torch.reshape(img_tensor, (b, -1, self.nums_hidden[-1], h_small, w_small)) # b,l,c_f,h,w
        encoderout = self.encoder(img_tensor) # b,l,c_f,h,w
        encoderout = torch.reshape(encoderout, (-1, self.nums_hidden[-1], h_small, w_small))

        raw_outputs = self.prediction(encoderout)
        raw_outputs = torch.reshape(raw_outputs, (-1, self.tot_raw_num, self.raw_channel_num, h, w))

        if self.use_flow:
            of_targets = of_targets_full

            input_of = input

            input_of = torch.reshape(input_of, (-1, self.raw_channel_num, h, w))

            img_tensor_of = self.feature_embedding_of(input_of)

            _, c_f, h_small, w_small = img_tensor_of.shape
            img_tensor_of = torch.reshape(img_tensor_of, (b, -1, self.nums_hidden[-1], h_small, w_small))  # b,l,c_f,h,w
            encoderout_of = self.encoder_of(img_tensor_of)  # b,l,c_f,h,w
            encoderout_of = torch.reshape(encoderout_of, (-1, self.nums_hidden[-1], h_small, w_small))

            of_outputs = self.prediction_of(encoderout_of)

            of_outputs = torch.reshape(of_outputs, (-1, self.tot_of_num, self.of_channel_num, h, w))

        else:
            of_outputs = []
            of_targets = []

        return of_outputs, raw_outputs, of_targets, raw_targets


class Unet(nn.Module):
    def  __init__(self, tot_raw_num, nums_hidden, use_flow=True):
        super(Unet, self).__init__()

        self.use_flow=use_flow
        self.tot_raw_num = tot_raw_num
        self.tot_of_num = tot_raw_num

        self.raw_channel_num = 3
        self.of_channel_num = 2

        self.inc = inconv(3, nums_hidden[0])
        self.down1 = down(nums_hidden[0], nums_hidden[1])
        self.down2 = down(nums_hidden[1], nums_hidden[2])

        self.up1 = up_unet(nums_hidden[2], nums_hidden[1])
        self.up2 = up_unet(nums_hidden[1], nums_hidden[0])
        self.out = outconv(nums_hidden[0], self.raw_channel_num)

        #of
        if self.use_flow:
            self.inc_of = inconv(3, nums_hidden[0])
            self.down1_of = down(nums_hidden[0], nums_hidden[1])
            self.down2_of = down(nums_hidden[1], nums_hidden[2])

            self.up1_of = up_unet(nums_hidden[2], nums_hidden[1])
            self.up2_of = up_unet(nums_hidden[1], nums_hidden[0])
            self.out_of = outconv(nums_hidden[0], self.of_channel_num)

    def forward(self, input, of_targets_full):


        b,c_in,h,w = input.shape
        assert c_in == self.raw_channel_num*self.tot_raw_num

        # convert to 5 dimensions for inputs
        input = input.permute(0, 2, 3, 1).contiguous() # b,h,w,c_in
        new_shape_input = input.size()[:-1] + (self.raw_channel_num, self.tot_raw_num) # b,h,w,c,l
        input = input.view(*new_shape_input)
        input = input.permute(0, 4, 3, 1, 2).contiguous().cuda() # b,l,c,h,w

        of_targets_full = of_targets_full.permute(0, 2, 3, 1).contiguous()
        new_shape_of_targets = of_targets_full.size()[:-1] + (self.of_channel_num, self.tot_of_num)
        of_targets_full = of_targets_full.view(*new_shape_of_targets)
        of_targets_full = of_targets_full.permute(0, 4, 3, 1, 2).contiguous().cuda()


        # interpolation
        input_frames = input
        raw_targets = input  # [...,1:]
        input_frames = torch.reshape(input_frames, (-1, self.raw_channel_num, h, w))
        out_1 = self.inc(input_frames)
        out_2 = self.down1(out_1)
        out_3 = self.down2(out_2)

        raw_outputs = self.up1(out_3, out_2)
        raw_outputs = self.up2(raw_outputs, out_1)
        raw_outputs = self.out(raw_outputs)
        raw_outputs = torch.reshape(raw_outputs, (-1, self.tot_raw_num, self.raw_channel_num, h, w))

        if self.use_flow:
            of_targets = of_targets_full

            input_of = input

            input_of = torch.reshape(input_of, (-1, self.raw_channel_num, h, w))
            out_1_of = self.inc_of(input_of)
            out_2_of = self.down1_of(out_1_of)
            out_3_of = self.down2_of(out_2_of)

            of_outputs = self.up1_of(out_3_of, out_2_of)
            of_outputs = self.up2_of(of_outputs, out_1_of)
            of_outputs = self.out_of(of_outputs)
            of_outputs = torch.reshape(of_outputs, (-1, self.tot_raw_num, self.of_channel_num, h, w))


        else:
            of_outputs = []
            of_targets = []


        return of_outputs, raw_outputs, of_targets, raw_targets

class Conv_LSTM(nn.Module):
    def __init__(self, tot_raw_num, nums_hidden, use_flow=True):
        super(Conv_LSTM, self).__init__()
        self.raw_channel_num = 3  # RGB channel no.
        self.of_channel_num = 2


        # self.feature_embedding = FeatureEmbedding(model_depth)
        self.feature_embedding = Spatial_Encoder(nums_hidden, self.raw_channel_num)

        self.prediction = Spatial_Decoder(nums_hidden, self.raw_channel_num)
        self.convlstm = LSTM(input_dim = nums_hidden[-1], hidden_dim=[nums_hidden[-1],nums_hidden[-1],nums_hidden[-1],
                                                                      nums_hidden[-1], nums_hidden[-1]],
                             kernel_size=(3,3), num_layers=5,
                            batch_first=True, bias=True, return_all_layers=False)

        if use_flow:
            self.feature_embedding_of = Spatial_Encoder(nums_hidden, self.raw_channel_num)

            self.convlstm_of = LSTM(input_dim=nums_hidden[-1],
                                 hidden_dim=[nums_hidden[-1], nums_hidden[-1], nums_hidden[-1]],
                                 kernel_size=(3, 3), num_layers=3,
                                 batch_first=True, bias=True, return_all_layers=False)

            self.prediction_of = Spatial_Decoder(nums_hidden, self.of_channel_num)

        self.tot_raw_num = tot_raw_num
        self.tot_of_num = tot_raw_num
        self.use_flow = use_flow
        self.nums_hidden = nums_hidden


    def forward(self, input, of_targets_full):


        b,c_in,h,w = input.shape
        assert c_in == self.raw_channel_num*self.tot_raw_num

        # convert to 5 dimensions for inputs
        input = input.permute(0, 2, 3, 1).contiguous() # b,h,w,c_in
        new_shape_input = input.size()[:-1] + (self.raw_channel_num, self.tot_raw_num) # b,h,w,c,l
        input = input.view(*new_shape_input)
        input = input.permute(0, 4, 3, 1, 2).contiguous().cuda() # b,l,c,h,w

        of_targets_full = of_targets_full.permute(0, 2, 3, 1).contiguous()
        new_shape_of_targets = of_targets_full.size()[:-1] + (self.of_channel_num, self.tot_of_num)
        of_targets_full = of_targets_full.view(*new_shape_of_targets)
        of_targets_full = of_targets_full.permute(0, 4, 3, 1, 2).contiguous().cuda()

        raw_targets = input
        input_frames = input
        input_frames = torch.reshape(input_frames, (-1, self.raw_channel_num, h, w))

        img_tensor = self.feature_embedding(input_frames) # b*l,c_f,h,w
        _, c_f, h_small, w_small = img_tensor.shape

        img_tensor = torch.reshape(img_tensor, (-1, self.tot_raw_num, self.nums_hidden[-1], h_small, w_small))

        img_tensor, _ = self.convlstm(img_tensor)
        # print(img_tensor[0].size())
        # zz
        # print(img_tensor[0][0].size())
        img_tensor = torch.reshape(img_tensor[0], (-1, self.nums_hidden[-1], h_small, w_small))

        raw_outputs = self.prediction(img_tensor)
        raw_outputs = torch.reshape(raw_outputs, (-1, self.tot_raw_num, self.raw_channel_num, h, w))


        if self.use_flow:
            of_targets = of_targets_full

            input_of = torch.reshape(input, (-1, self.raw_channel_num, h, w))

            img_tensor_of = self.feature_embedding_of(input_of)

            _, c_f, h_small, w_small = img_tensor_of.shape

            img_tensor_of = torch.reshape(img_tensor_of, (-1, self.tot_of_num, self.nums_hidden[-1], h_small, w_small))

            img_tensor_of, _ = self.convlstm_of(img_tensor_of)
            img_tensor_of = torch.reshape(img_tensor_of[0], (-1, self.nums_hidden[-1], h_small, w_small))

            of_outputs = self.prediction_of(img_tensor_of)

            of_outputs = torch.reshape(of_outputs, (-1, self.tot_of_num, self.of_channel_num, h, w))

        else:
            of_outputs = []
            of_targets = []


        return of_outputs, raw_outputs, of_targets, raw_targets



