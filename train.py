import numpy as np
import os
from torch.utils.data import DataLoader
from vad_datasets import unified_dataset_interface, cube_to_train_dataset

from vad_datasets import bbox_collate, img_tensor2numpy, img_batch_tensor2numpy, frame_size

from helper.misc import AverageMeter
import torch
from state_model import ConvTransformer_recon_correct
import torch.optim as optim
import torch.nn as nn
import argparse


import os
import sys

pyfile_name = os.path.basename(sys.argv[0]).split(".")[0]

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return  True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('boolean value expected')

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', default='avenue', type=str)
parser.add_argument('-n_l', '--num_layers', default=3, type=int)
parser.add_argument('-n_h', '--num_heads', default=4, type=int)
parser.add_argument('-pe', '--positional_encoding', default='learned', type=str)
parser.add_argument('-e', '--epochs', default=20, type=int)
parser.add_argument('-b', '--batch_size', default=128, type=int)
parser.add_argument('-l', '--temporal_length', default=3, type=int)
parser.add_argument('-lam_r', '--lambda_raw', default=1, type=float)
parser.add_argument('-lam_o', '--lambda_of', default=1, type=float)
parser.add_argument('-train_b', '--train_bbox_saved', default=True, type=str2bool)
parser.add_argument('-train_f', '--train_foreground_saved', default=True, type=str2bool)
parser.add_argument('-f', '--use_flow', default=True, type=str2bool)

parser.add_argument('-bd', '--border_mode', default='elastic', type=str)




args = parser.parse_args()


def calc_block_idx(x_min, x_max, y_min, y_max, h_step, w_step, mode):
    all_blocks = list()
    center = np.array([(y_min + y_max) / 2, (x_min + x_max) / 2])
    all_blocks.append(center + center)
    if mode > 1:
        all_blocks.append(np.array([y_min, center[1]]) + center)
        all_blocks.append(np.array([y_max, center[1]]) + center)
        all_blocks.append(np.array([center[0], x_min]) + center)
        all_blocks.append(np.array([center[0], x_max]) + center)
    if mode >= 9:
        all_blocks.append(np.array([y_min, x_min]) + center)
        all_blocks.append(np.array([y_max, x_max]) + center)
        all_blocks.append(np.array([y_max, x_min]) + center)
        all_blocks.append(np.array([y_min, x_max]) + center)
    all_blocks = np.array(all_blocks) / 2
    h_block_idxes = all_blocks[:, 0] / h_step
    w_block_idxes = all_blocks[:, 1] / w_step
    h_block_idxes, w_block_idxes = list(h_block_idxes.astype(np.int)), list(w_block_idxes.astype(np.int))
    # delete repeated elements
    all_blocks = set([x for x in zip(h_block_idxes, w_block_idxes)])
    all_blocks = [x for x in all_blocks]
    return all_blocks


#  /*------------------------------------overall parameter setting------------------------------------------*/

dataset_name = args.dataset
raw_dataset_dir = 'raw_datasets'
foreground_extraction_mode = 'obj_det_with_motion'
data_root_dir = 'data'
modality = 'raw2flow'
mode ='train'
method = 'SelfComplete'
num_layers = args.num_layers
num_heads = args.num_heads
pe = args.positional_encoding
context_frame_num = args.temporal_length
context_of_num = args.temporal_length

patch_size = 32
h_block = 1
w_block = 1
train_block_mode = 1
bbox_saved = args.train_bbox_saved
foreground_saved = args.train_foreground_saved
motionThr = 0


#  /*------------------------------------------foreground extraction----------------------------------------------*/
config_file = './obj_det_config/cascade_rcnn_r101_fpn_1x.py'
checkpoint_file = './obj_det_checkpoints/cascade_rcnn_r101_fpn_1x_20181129-d64ebac7.pth'

# set dataset for foreground extraction
dataset = unified_dataset_interface(dataset_name=dataset_name, dir=os.path.join(raw_dataset_dir, dataset_name),
                                    context_frame_num=1, mode=mode, border_mode='hard')

if not bbox_saved:
    # build the model from a config file and a checkpoint file
    from fore_det.inference import init_detector
    from fore_det.obj_det_with_motion import imshow_bboxes, getObBboxes, getFgBboxes, delCoverBboxes
    from fore_det.simple_patch import get_patch_loc
    
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    collate_func = bbox_collate('train')
    dataset_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1,
                                collate_fn=collate_func.collate)
    all_bboxes = list()

    for idx in range(dataset.__len__()):
        batch, _ = dataset.__getitem__(idx)
        print('Extracting bboxes of {}-th frame'.format(idx + 1))
        cur_img = img_tensor2numpy(batch[1])

        if foreground_extraction_mode == 'obj_det_with_motion':
            # A coarse detection of bboxes by pretrained object detector
            ob_bboxes = getObBboxes(cur_img, model, dataset_name)

            ob_bboxes = delCoverBboxes(ob_bboxes, dataset_name)

            # further foreground detection by motion
            fg_bboxes = getFgBboxes(cur_img, img_batch_tensor2numpy(batch), ob_bboxes, dataset_name, verbose=False)

            if fg_bboxes.shape[0] > 0:
                cur_bboxes = np.concatenate((ob_bboxes, fg_bboxes), axis=0)
            else:
                cur_bboxes = ob_bboxes
        elif foreground_extraction_mode == 'obj_det':
            # A coarse detection of bboxes by pretrained object detector
            ob_bboxes = getObBboxes(cur_img, model, dataset_name)
            cur_bboxes = delCoverBboxes(ob_bboxes, dataset_name)
        elif foreground_extraction_mode == 'simple_patch':
            patch_num_list = [(3, 4), (6, 8)]
            cur_bboxes = list()
            for h_num, w_num in patch_num_list:
                cur_bboxes.append(get_patch_loc(frame_size[dataset_name][0], frame_size[dataset_name][1], h_num, w_num))
            cur_bboxes = np.concatenate(cur_bboxes, axis=0)
        else:
            raise NotImplementedError

        # imshow_bboxes(cur_img, cur_bboxes)
        all_bboxes.append(cur_bboxes)
    np.save(os.path.join(dataset.dir, 'bboxes_train_{}.npy'.format(foreground_extraction_mode)), all_bboxes)
    print('bboxes for training data saved!')
else:
    all_bboxes = np.load(os.path.join(dataset.dir, 'bboxes_train_{}.npy'.format(foreground_extraction_mode)),
                         allow_pickle=True)
    print('bboxes for training data loaded!')

# /------------------------- extract foreground using extracted bboxes---------------------------------------/



border_mode = args.border_mode

if not foreground_saved:
    if modality == 'raw_datasets':
        file_format = frame_size[dataset_name][2]
    elif modality == 'raw2flow':
        file_format1 = frame_size[dataset_name][2]
        file_format2 = '.npy'
    else:
        file_format = '.npy'

    if modality == 'raw2flow':
        dataset = unified_dataset_interface(dataset_name=dataset_name, dir=os.path.join('raw_datasets', dataset_name),
                                            context_frame_num=context_frame_num, mode=mode, border_mode=border_mode,
                                            all_bboxes=all_bboxes, patch_size=patch_size, file_format=file_format1)

        dataset2 = unified_dataset_interface(dataset_name=dataset_name, dir=os.path.join('optical_flow', dataset_name),
                                         context_frame_num=context_of_num, mode=mode, border_mode=border_mode,
                                         all_bboxes=all_bboxes, patch_size=patch_size, file_format=file_format2)
    else:
        dataset = unified_dataset_interface(dataset_name=dataset_name, dir=os.path.join(modality, dataset_name),
                                            context_frame_num=context_frame_num, mode=mode, border_mode=border_mode,
                                            all_bboxes=all_bboxes, patch_size=patch_size, file_format=file_format)

    if dataset_name == 'ShanghaiTech':
        foreground_set = [[[[] for ww in range(w_block)] for hh in range(h_block)] for ss in range(dataset.scene_num)]
        if modality == 'raw2flow':
            foreground_set2 = [[[[] for ww in range(w_block)] for hh in range(h_block)] for ss in
                               range(dataset.scene_num)]
    else:
        foreground_set = [[[] for ww in range(w_block)] for hh in range(h_block)]
        if modality == 'raw2flow':
            foreground_set2 = [[[] for ww in range(w_block)] for hh in range(h_block)]

    h_step, w_step = frame_size[dataset_name][0] / h_block, frame_size[dataset_name][1] / w_block
    dataset_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1,
                                collate_fn=bbox_collate(mode=mode).collate)

    if dataset_name == 'ShanghaiTech' and modality == 'raw2flow':
        randIdx = np.random.permutation(dataset.__len__())
        cout = 0
        segIdx = 0
        saveSegNum = 40000

    for iidx in range(dataset.__len__()):
        if dataset_name == 'ShanghaiTech' and modality == 'raw2flow':
            idx = randIdx[iidx]
            cout += 1
        else:
            idx = iidx

        batch, _ = dataset.__getitem__(idx)
        if modality == 'raw2flow':
            batch2, _ = dataset2.__getitem__(idx)

        if dataset_name == 'ShanghaiTech':
            print(
                'Extracting foreground in {}-th batch, {} in total, scene: {}'.format(iidx + 1, dataset.__len__() // 1,
                                                                                      dataset.scene_idx[idx]))
        else:
            print('Extracting foreground in {}-th batch, {} in total'.format(iidx + 1, dataset.__len__() // 1))

        cur_bboxes = all_bboxes[idx]
        if len(cur_bboxes) > 0:
            batch = img_batch_tensor2numpy(batch)
            if modality == 'raw2flow':
                batch2 = img_batch_tensor2numpy(batch2)

            if modality == 'optical_flow':
                if len(batch.shape) == 4:
                    mag = np.sum(np.sum(np.sum(batch ** 2, axis=3), axis=2), axis=1)
                else:
                    mag = np.mean(np.sum(np.sum(np.sum(batch ** 2, axis=4), axis=3), axis=2), axis=1)
            elif modality == 'raw2flow':
                if len(batch2.shape) == 4:
                    mag = np.sum(np.sum(np.sum(batch2 ** 2, axis=3), axis=2), axis=1)
                else:
                    mag = np.mean(np.sum(np.sum(np.sum(batch2 ** 2, axis=4), axis=3), axis=2), axis=1)
            else:
                mag = np.ones(batch.shape[0]) * 10000

            for idx_bbox in range(cur_bboxes.shape[0]):
                if mag[idx_bbox] > motionThr:
                    all_blocks = calc_block_idx(cur_bboxes[idx_bbox, 0], cur_bboxes[idx_bbox, 2],
                                                cur_bboxes[idx_bbox, 1], cur_bboxes[idx_bbox, 3], h_step, w_step,
                                                mode=train_block_mode)
                    for (h_block_idx, w_block_idx) in all_blocks:
                        if dataset_name == 'ShanghaiTech':
                            foreground_set[dataset.scene_idx[idx] - 1][h_block_idx][w_block_idx].append(batch[idx_bbox])
                            if modality == 'raw2flow':
                                foreground_set2[dataset.scene_idx[idx] - 1][h_block_idx][w_block_idx].append(
                                    batch2[idx_bbox])
                        else:
                            foreground_set[h_block_idx][w_block_idx].append(batch[idx_bbox])
                            if modality == 'raw2flow':
                                foreground_set2[h_block_idx][w_block_idx].append(batch2[idx_bbox])

        if dataset_name == 'ShanghaiTech' and modality == 'raw2flow':
            if cout == saveSegNum:
                foreground_set = [
                    [[np.array(foreground_set[ss][hh][ww]) for ww in range(w_block)] for hh in range(h_block)] for ss in
                    range(dataset.scene_num)]
                foreground_set2 = [
                    [[np.array(foreground_set2[ss][hh][ww]) for ww in range(w_block)] for hh in range(h_block)] for ss
                    in range(dataset.scene_num)]
                np.save(os.path.join(data_root_dir, modality,
                                     dataset_name + '_' + 'foreground_train_{}_seg_{}_{}_border_{}-raw.npy'.format(
                                         foreground_extraction_mode, segIdx, context_frame_num, border_mode)), foreground_set)
                np.save(os.path.join(data_root_dir, modality,
                                     dataset_name + '_' + 'foreground_train_{}_seg_{}_{}_border_{}-flow.npy'.format(
                                         foreground_extraction_mode, segIdx, context_frame_num, border_mode)), foreground_set2)
                del foreground_set, foreground_set2

                cout = 0
                segIdx += 1
                foreground_set = [[[[] for ww in range(w_block)] for hh in range(h_block)] for ss in
                                  range(dataset.scene_num)]
                foreground_set2 = [[[[] for ww in range(w_block)] for hh in range(h_block)] for ss in
                                   range(dataset.scene_num)]

    if dataset_name == 'ShanghaiTech':
        if modality != 'raw2flow':
            foreground_set = [[[np.array(foreground_set[ss][hh][ww]) for ww in range(w_block)] for hh in range(h_block)]
                              for ss in range(dataset.scene_num)]
            np.save(os.path.join(data_root_dir, modality,
                                 dataset_name + '_' + 'foreground_train_{}.npy'.format(foreground_extraction_mode)),
                    foreground_set)
        else:
            if dataset.__len__() % saveSegNum != 0:
                foreground_set = [
                    [[np.array(foreground_set[ss][hh][ww]) for ww in range(w_block)] for hh in range(h_block)]
                    for ss in range(dataset.scene_num)]
                foreground_set2 = [
                    [[np.array(foreground_set2[ss][hh][ww]) for ww in range(w_block)] for hh in range(h_block)] for ss
                    in
                    range(dataset.scene_num)]
                np.save(os.path.join(data_root_dir, modality,
                                     dataset_name + '_' + 'foreground_train_{}_seg_{}_{}_border_{}-raw.npy'.format(
                                         foreground_extraction_mode, segIdx, context_frame_num, border_mode)), foreground_set)
                np.save(os.path.join(data_root_dir, modality,
                                     dataset_name + '_' + 'foreground_train_{}_seg_{}_{}_border_{}-flow.npy'.format(
                                         foreground_extraction_mode, segIdx, context_frame_num, border_mode)), foreground_set2)
    else:
        if modality == 'raw2flow':
            foreground_set = [[np.array(foreground_set[hh][ww]) for ww in range(w_block)] for hh in range(h_block)]
            np.save(os.path.join(data_root_dir, modality,
                                 dataset_name + '_' + 'foreground_train_{}_{}_border_{}-raw.npy'.format(foreground_extraction_mode, context_frame_num, border_mode)),
                    foreground_set)
            foreground_set2 = [[np.array(foreground_set2[hh][ww]) for ww in range(w_block)] for hh in range(h_block)]
            np.save(os.path.join(data_root_dir, modality, dataset_name + '_' + 'foreground_train_{}_{}_border_{}-flow.npy'.format(
                foreground_extraction_mode, context_frame_num, border_mode)), foreground_set2)
        else:
            foreground_set = [[np.array(foreground_set[hh][ww]) for ww in range(w_block)] for hh in range(h_block)]
            np.save(os.path.join(data_root_dir, modality,
                                 dataset_name + '_' + 'foreground_train_{}_{}_border_{}.npy'.format(foreground_extraction_mode, context_frame_num, border_mode)),
                    foreground_set)
    print('foreground for training data saved!')
else:
    if dataset_name != 'ShanghaiTech':
        if modality == 'raw2flow':
            foreground_set = np.load(os.path.join(data_root_dir, modality,
                                                  dataset_name + '_' + 'foreground_train_{}_{}_border_{}-raw.npy'.format(
                                                      foreground_extraction_mode, context_frame_num, border_mode)), allow_pickle=True)
            foreground_set2 = np.load(os.path.join(data_root_dir, modality,
                                                   dataset_name + '_' + 'foreground_train_{}_{}_border_{}-flow.npy'.format(
                                                       foreground_extraction_mode, context_frame_num, border_mode)), allow_pickle=True)
        else:
            foreground_set = np.load(os.path.join(data_root_dir, modality,
                                                  dataset_name + '_' + 'foreground_train_{}_{}_border_{}.npy'.format(
                                                      foreground_extraction_mode, context_frame_num, border_mode)), allow_pickle=True)
        print('foreground for training data loaded!')
    else:
        if modality != 'raw2flow':
            foreground_set = np.load(os.path.join(data_root_dir, modality,
                                                  dataset_name + '_' + 'foreground_train_{}_{}_border_{}.npy'.format(
                                                      foreground_extraction_mode, context_frame_num, border_mode)), allow_pickle=True)

#  /*------------------------------------------Normal event modeling----------------------------------------------*/


if method == 'SelfComplete':
    loss_func = nn.MSELoss()
    epochs = args.epochs
    batch_size = args.batch_size
    useFlow = args.use_flow
    if border_mode == 'predict':
        tot_frame_num = context_frame_num + 1
        tot_of_num = context_of_num + 1
    else:
        tot_frame_num = 2 * context_frame_num + 1
        tot_of_num = 2 * context_of_num + 1
    rawRange = 10
    if rawRange >= tot_frame_num:  # if rawRange is out of the range, use all frames
        rawRange = None
    padding = False
    lambda_raw = args.lambda_raw
    lambda_of = args.lambda_of

    assert modality == 'raw2flow'
    if dataset_name == 'ShanghaiTech':
        model_set = [[[[] for ww in range(w_block)] for hh in range(h_block)] for ss in
                     range(frame_size[dataset_name][-1])]
        raw_training_scores_set = [[[[] for ww in range(w_block)] for hh in range(h_block)] for ss in
                                   range(frame_size[dataset_name][-1])]
        of_training_scores_set = [[[[] for ww in range(w_block)] for hh in range(h_block)] for ss in
                                  range(frame_size[dataset_name][-1])]
    else:
        model_set = [[[] for ww in range(len(foreground_set[hh]))] for hh in range(len(foreground_set))]
        raw_training_scores_set = [[[] for ww in range(len(foreground_set[hh]))] for hh in range(len(foreground_set))]
        of_training_scores_set = [[[] for ww in range(len(foreground_set[hh]))] for hh in range(len(foreground_set))]

    # Prepare training data in current block
    if dataset_name == 'ShanghaiTech':
        saveSegNum = 40000
        totSegNum = np.int(np.ceil(dataset.__len__() / saveSegNum))
        for s_idx in range(len(model_set)):
            for h_idx in range(len(model_set[s_idx])):
                for w_idx in range(len(model_set[s_idx][h_idx])):
                    raw_losses = AverageMeter()
                    of_losses = AverageMeter()
                    cur_model = torch.nn.DataParallel(ConvTransformer_recon_correct(
                                                     tot_raw_num = tot_frame_num, nums_hidden = [32, 64, 128], num_layers=num_layers,
                                                     num_dec_frames=1, num_heads=num_heads, with_residual=True,
                                                     with_pos=True, pos_kind=pe, mode=0, use_flow=useFlow)).cuda()
                    optimizer = optim.Adam(cur_model.parameters(), eps=1e-7, weight_decay=0.000)
                    cur_model.train()
                    for epoch in range(epochs):
                        for segIdx in range(totSegNum):
                            foreground_set = np.load(os.path.join(data_root_dir, modality,
                                                                  dataset_name + '_' + 'foreground_train_{}_seg_{}_{}_border_{}-raw.npy'.format(
                                                                      foreground_extraction_mode, segIdx, context_frame_num, border_mode)))
                            foreground_set2 = np.load(os.path.join(data_root_dir, modality,
                                                                   dataset_name + '_' + 'foreground_train_{}_seg_{}_{}_border_{}-flow.npy'.format(
                                                                       foreground_extraction_mode, segIdx, context_frame_num, border_mode)))
                            cur_training_data = foreground_set[s_idx][h_idx][w_idx]
                            cur_training_data2 = foreground_set2[s_idx][h_idx][w_idx]
                            cur_dataset = cube_to_train_dataset(cur_training_data, target=cur_training_data2)
                            cur_dataloader = DataLoader(dataset=cur_dataset, batch_size=batch_size, shuffle=True)

                            for idx, (inputs, of_targets_all, _) in enumerate(cur_dataloader):
                                inputs = inputs.cuda().type(torch.cuda.FloatTensor)
                                of_targets_all = of_targets_all.cuda().type(torch.cuda.FloatTensor)

                                of_outputs, raw_outputs, of_targets, raw_targets = cur_model(inputs, of_targets_all)

                                loss_raw = loss_func(raw_targets.detach(), raw_outputs)
                                if useFlow:
                                    loss_of = loss_func(of_targets.detach(), of_outputs)

                                if useFlow:
                                    loss = lambda_raw * loss_raw + lambda_of * loss_of
                                else:
                                    loss = loss_raw

                                raw_losses.update(loss_raw.item(), inputs.size(0))
                                if useFlow:
                                    of_losses.update(loss_of.item(), inputs.size(0))
                                else:
                                    of_losses.update(0., inputs.size(0))

                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()

                                if idx % 5 == 0:
                                    print(
                                        'Block: ({}, {}), epoch {}, seg {}, batch {} of {}, raw loss: {}, of loss: {}'.format(
                                            h_idx, w_idx, epoch, segIdx, idx, cur_dataset.__len__() // batch_size,
                                            raw_losses.avg,
                                            of_losses.avg))

                        #         break
                        #     break
                        # break

                    model_set[s_idx][h_idx][w_idx].append(cur_model.state_dict())

                    #  /*--  A forward pass to store the training scores of optical flow and raw datasets respectively*/
                    for segIdx in range(totSegNum):
                        foreground_set = np.load(os.path.join(data_root_dir, modality,
                                                              dataset_name + '_' + 'foreground_train_{}_seg_{}_{}_border_{}-raw.npy'.format(
                                                                  foreground_extraction_mode, segIdx, context_frame_num, border_mode)))
                        foreground_set2 = np.load(os.path.join(data_root_dir, modality,
                                                               dataset_name + '_' + 'foreground_train_{}_seg_{}_{}_border_{}-flow.npy'.format(
                                                                   foreground_extraction_mode, segIdx, context_frame_num, border_mode)))
                        cur_training_data = foreground_set[s_idx][h_idx][w_idx]
                        cur_training_data2 = foreground_set2[s_idx][h_idx][w_idx]
                        cur_dataset = cube_to_train_dataset(cur_training_data, target=cur_training_data2)

                        forward_dataloader = DataLoader(dataset=cur_dataset, batch_size=batch_size//4, shuffle=False)
                        score_func = nn.MSELoss(reduce=False)
                        cur_model.eval()
                        for idx, (inputs, of_targets_all, _) in enumerate(forward_dataloader):
                            inputs = inputs.cuda().type(torch.cuda.FloatTensor)
                            of_targets_all = of_targets_all.cuda().type(torch.cuda.FloatTensor)

                            of_outputs, raw_outputs, of_targets, raw_targets = cur_model(inputs, of_targets_all)
                            raw_scores = score_func(raw_targets, raw_outputs).cpu().data.numpy()
                            raw_scores = np.sum(np.sum(np.sum(np.sum(raw_scores, axis=4), axis=3), axis=2), axis=1)  # mse
                            raw_training_scores_set[s_idx][h_idx][w_idx].append(raw_scores)
                            if useFlow:
                                of_scores = score_func(of_targets, of_outputs).cpu().data.numpy()
                                of_scores = np.sum(np.sum(np.sum(np.sum(of_scores, axis=4), axis=3), axis=2), axis=1)  # mse
                                of_training_scores_set[s_idx][h_idx][w_idx].append(of_scores)

                    raw_training_scores_set[s_idx][h_idx][w_idx] = np.concatenate(
                        raw_training_scores_set[s_idx][h_idx][w_idx], axis=0)
                    if useFlow:
                        of_training_scores_set[s_idx][h_idx][w_idx] = np.concatenate(
                            of_training_scores_set[s_idx][h_idx][w_idx], axis=0)
                    del cur_model, raw_losses, of_losses

        torch.save(raw_training_scores_set, os.path.join(data_root_dir, modality,
                                                         dataset_name + '_' + 'raw_training_scores_border_{}_head_{}_layer_{}_length_{}_pe_{}_epoch_{}_lambda_{}_{}'.format(
                                                             border_mode, num_heads, num_layers, context_frame_num, pe,
                                                             epochs, lambda_raw, lambda_of) + '_' + 'pyname_{}.npy'.format(pyfile_name)))
        torch.save(of_training_scores_set, os.path.join(data_root_dir, modality,
                                                        dataset_name + '_' + 'of_training_scores_border_{}_head_{}_layer_{}_length_{}_pe_{}_epoch_{}_lambda_{}_{}'.format(
                                                            border_mode, num_heads, num_layers, context_frame_num, pe,
                                                            epochs, lambda_raw, lambda_of) + '_' + 'pyname_{}.npy'.format(pyfile_name)))
    else:
        raw_losses = AverageMeter()
        of_losses = AverageMeter()
        torch.autograd.set_detect_anomaly(True)
        for h_idx in range(len(foreground_set)):
            for w_idx in range(len(foreground_set[h_idx])):
                cur_training_data = foreground_set[h_idx][w_idx]

                if len(cur_training_data) > 1:  # num > 1 for data parallel
                    cur_training_data2 = foreground_set2[h_idx][w_idx]
                    cur_dataset = cube_to_train_dataset(cur_training_data, target=cur_training_data2)
                    cur_dataloader = DataLoader(dataset=cur_dataset, batch_size=batch_size, shuffle=True)

                    cur_model = torch.nn.DataParallel(ConvTransformer_recon_correct(
                                                     tot_raw_num = tot_frame_num, nums_hidden = [32, 64, 128], num_layers=num_layers,
                                                     num_dec_frames=1, num_heads=num_heads, with_residual=True,
                                                     with_pos=True, pos_kind=pe, mode=0, use_flow=useFlow)).cuda()
                    if dataset_name == 'UCSDped2':
                        optimizer = optim.Adam(cur_model.parameters(), eps=1e-7, weight_decay=0.0)
                    else:
                        optimizer = optim.Adam(cur_model.parameters(), eps=1e-7, weight_decay=0.0)

                    cur_model.train()
                    for epoch in range(epochs):
                        for idx, (inputs, of_targets_all, _) in enumerate(cur_dataloader):
                            inputs = inputs.cuda().type(torch.cuda.FloatTensor)
                            # print(torch.max(inputs))
                            of_targets_all = of_targets_all.cuda().type(torch.cuda.FloatTensor)

                            of_outputs, raw_outputs, of_targets, raw_targets = cur_model(inputs, of_targets_all)

                            loss_raw = loss_func(raw_targets.detach(), raw_outputs)
                            if useFlow:
                                loss_of = loss_func(of_targets.detach(), of_outputs)

                            if useFlow:
                                loss = lambda_raw * loss_raw + lambda_of * loss_of
                            else:
                                loss = loss_raw

                            raw_losses.update(loss_raw.item(), inputs.size(0))
                            if useFlow:
                                of_losses.update(loss_of.item(), inputs.size(0))
                            else:
                                of_losses.update(0., inputs.size(0))

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            if idx % 5 == 0:
                                max_num = 20
                                print(
                                    'Block: ({}, {}), epoch {}, batch {} of {}, raw loss: {}, of loss: {}'.format(h_idx,
                                                                                                                  w_idx,
                                                                                                                  epoch,
                                                                                                                  idx,
                                                                                                                  cur_dataset.__len__() // batch_size,
                                                                                                                  raw_losses.avg,
                                                                                                                  of_losses.avg))

                    model_set[h_idx][w_idx].append(cur_model.state_dict())

                    #  /*--  A forward pass to store the training scores of optical flow and raw datasets respectively*/
                    forward_dataloader = DataLoader(dataset=cur_dataset, batch_size=batch_size//4, shuffle=False)
                    # raw_score_func = nn.MSELoss(reduce=False)
                    # of_score_func = nn.L1Loss(reduce=False)
                    score_func = nn.MSELoss(reduce=False)
                    cur_model.eval()
                    for idx, (inputs, of_targets_all, _) in enumerate(forward_dataloader):
                        inputs = inputs.cuda().type(torch.cuda.FloatTensor)
                        of_targets_all = of_targets_all.cuda().type(torch.cuda.FloatTensor)
                        of_outputs, raw_outputs, of_targets, raw_targets = cur_model(inputs, of_targets_all)

                        raw_scores = score_func(raw_targets, raw_outputs).cpu().data.numpy()
                        raw_scores = np.sum(np.sum(np.sum(np.sum(raw_scores, axis=4), axis=3), axis=2), axis=1)  # mse
                        raw_training_scores_set[h_idx][w_idx].append(raw_scores)
                        if useFlow:
                            of_scores = score_func(of_targets, of_outputs).cpu().data.numpy()
                            of_scores = np.sum(np.sum(np.sum(np.sum(of_scores, axis=4), axis=3), axis=2), axis=1)  # mse
                            of_training_scores_set[h_idx][w_idx].append(of_scores)

                    raw_training_scores_set[h_idx][w_idx] = np.concatenate(raw_training_scores_set[h_idx][w_idx], axis=0)
                    if useFlow:
                        of_training_scores_set[h_idx][w_idx] = np.concatenate(of_training_scores_set[h_idx][w_idx],
                                                                              axis=0)
        torch.save(raw_training_scores_set, os.path.join(data_root_dir, modality,
                                                         dataset_name + '_' + 'raw_training_scores_border_{}_head_{}_layer_{}_length_{}_pe_{}_epoch_{}_lambda_{}_{}'.format(
                                                              border_mode, num_heads, num_layers, context_frame_num, pe, epochs, lambda_raw, lambda_of) + '_' + 'pyname_{}.npy'.format(pyfile_name)))
        torch.save(of_training_scores_set, os.path.join(data_root_dir, modality,
                                                        dataset_name + '_' + 'of_training_scores_border_{}_head_{}_layer_{}_length_{}_pe_{}_epoch_{}_lambda_{}_{}'.format(
                                                             border_mode, num_heads, num_layers, context_frame_num, pe, epochs, lambda_raw, lambda_of) + '_' + 'pyname_{}.npy'.format(pyfile_name)))
        print('training scores saved')

    torch.save(model_set, os.path.join(data_root_dir, modality,
                                       dataset_name + '_' + 'model_{}_head_{}_layer_{}_length_{}_pe_{}_epoch_{}_lambda_{}_{}'.format(
                                           border_mode, num_heads, num_layers, context_frame_num, pe, epochs, lambda_raw, lambda_of) + '_' + 'pyname_{}.npy'.format(pyfile_name)))
    print('Training of {} for dataset: {} has completed!'.format(method, dataset_name))
else:
    raise NotImplementedError

