from xml.sax.xmlreader import InputSource
import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from vad_datasets import unified_dataset_interface

from vad_datasets import bbox_collate, img_tensor2numpy, img_batch_tensor2numpy, frame_size, cube_to_train_dataset


from state_model import ConvTransformer_recon_correct
import torch.nn as nn
from utils import save_roc_pr_curve_data
import time
import argparse


import os
import sys
# from helper.visualization_helper import visualize_pair, visualize_batch, visualize_recon, visualize_pair_map

pyfile_name = "train"
pyfile_name_score = os.path.basename(sys.argv[0]).split(".")[0]


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
parser.add_argument('-d', '--dataset', default='UCSDped2', type=str)
parser.add_argument('-n_l', '--num_layers', default=3, type=int)
parser.add_argument('-n_h', '--num_heads', default=4, type=int)
parser.add_argument('-pe', '--positional_encoding', default='learned', type=str)
parser.add_argument('-e', '--epochs', default=20, type=int)
parser.add_argument('-b', '--batch_size', default=128, type=int)
parser.add_argument('-l', '--temporal_length', default=3, type=int)
parser.add_argument('-lam_r', '--lambda_raw', default=1, type=float)
parser.add_argument('-lam_o', '--lambda_of', default=1, type=float)
parser.add_argument('-w_r', '--w_raw', default=1, type=float)
parser.add_argument('-w_o', '--w_of', default=1, type=float)
parser.add_argument('-test_b', '--test_bbox_saved', type=str2bool, default=True)
parser.add_argument('-test_f', '--test_foreground_saved', type=str2bool, default=True)
parser.add_argument('-f', '--use_flow', default=True, type=str2bool)
parser.add_argument('-s', '--scores_saved', default=False, type=str2bool)
parser.add_argument('-ep', '--epsilon', default=0.01, type=float)



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
mode ='test'
method = 'SelfComplete'
num_layers = args.num_layers
num_heads = args.num_heads
pe = args.positional_encoding
context_frame_num = args.temporal_length
context_of_num = args.temporal_length

patch_size = 32
h_block = 1
w_block = 1
test_block_mode = 1
bbox_saved = args.test_bbox_saved
foreground_saved = args.test_foreground_saved
motionThr = 0
epochs = args.epochs
# visual_save_dir = args.save_dir


#  /*------------------------------------------foreground extraction----------------------------------------------*/
config_file = './obj_det_config/cascade_rcnn_r101_fpn_1x.py'
checkpoint_file = './obj_det_checkpoints/cascade_rcnn_r101_fpn_1x_20181129-d64ebac7.pth'

# set dataset for foreground extraction
dataset = unified_dataset_interface(dataset_name=dataset_name, dir=os.path.join(raw_dataset_dir, dataset_name),
                                    context_frame_num=1, mode=mode, border_mode='hard')

if not bbox_saved:
    from fore_det.inference import init_detector

    from fore_det.obj_det_with_motion import imshow_bboxes, getObBboxes, getFgBboxes, delCoverBboxes
    from fore_det.simple_patch import get_patch_loc
    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    collate_func = bbox_collate('test')
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

        all_bboxes.append(cur_bboxes)
    np.save(os.path.join(dataset.dir, 'bboxes_test_{}.npy'.format(foreground_extraction_mode)), all_bboxes)
    print('bboxes for testing data saved!')
else:
    all_bboxes = np.load(os.path.join(dataset.dir, 'bboxes_test_{}.npy'.format(foreground_extraction_mode)),
                         allow_pickle=True)
    print('bboxes for testing data loaded!')

# /------------------------- extract foreground using extracted bboxes---------------------------------------/
# set dataset for foreground bbox extraction
if method == 'SelfComplete':
    border_mode = 'elastic'
else:
    border_mode = 'hard'

if not foreground_saved:

    if modality == 'raw_datasets':
        file_format = frame_size[dataset_name][2]
    elif modality == 'raw2flow':
        file_format1 = frame_size[dataset_name][2]
        file_format2 = '.npy'
    else:
        file_format = '.npy'

    # set dataset for foreground bbox extraction
    if modality == 'raw2flow':
        dataset = unified_dataset_interface(dataset_name=dataset_name, dir=os.path.join('raw_datasets', dataset_name),
                                            context_frame_num=context_frame_num, mode=mode,
                                            border_mode=border_mode, all_bboxes=all_bboxes, patch_size=patch_size,
                                            file_format=file_format1)
      
        dataset2 = unified_dataset_interface(dataset_name=dataset_name, dir=os.path.join('optical_flow', dataset_name),
                                             context_frame_num=context_of_num, mode=mode,
                                             border_mode=border_mode, all_bboxes=all_bboxes, patch_size=patch_size,
                                             file_format=file_format2)
    else:
        dataset = unified_dataset_interface(dataset_name=dataset_name, dir=os.path.join(modality, dataset_name),
                                            context_frame_num=context_frame_num, mode=mode,
                                            border_mode=border_mode, all_bboxes=all_bboxes, patch_size=patch_size,
                                            file_format=file_format)

    if dataset_name == 'ShanghaiTech':
        np.save(os.path.join(data_root_dir, modality, dataset_name + '_' + 'scene_idx.npy'), dataset.scene_idx)
        scene_idx = dataset.scene_idx

    foreground_set = [[[[] for ww in range(w_block)] for hh in range(h_block)] for ii in range(dataset.__len__())]
    if modality == 'raw2flow':
        foreground_set2 = [[[[] for ww in range(w_block)] for hh in range(h_block)] for ii in range(dataset.__len__())]
    foreground_bbox_set = [[[[] for ww in range(w_block)] for hh in range(h_block)] for ii in range(dataset.__len__())]
    h_step, w_step = frame_size[dataset_name][0] / h_block, frame_size[dataset_name][1] / w_block
    dataset_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1,
                                collate_fn=bbox_collate(mode=mode).collate)

    for idx in range(dataset.__len__()):
        batch, _ = dataset.__getitem__(idx)
        if modality == 'raw2flow':
            batch2, _ = dataset2.__getitem__(idx)
        print('Extracting foreground in {}-th batch, {} in total'.format(idx + 1, dataset.__len__() // 1))
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
                                                mode=test_block_mode)
                    for (h_block_idx, w_block_idx) in all_blocks:
                        foreground_set[idx][h_block_idx][w_block_idx].append(batch[idx_bbox])
                        if modality == 'raw2flow':
                            foreground_set2[idx][h_block_idx][w_block_idx].append(batch2[idx_bbox])
                        foreground_bbox_set[idx][h_block_idx][w_block_idx].append(cur_bboxes[idx_bbox])

    foreground_set = [[[np.array(foreground_set[ii][hh][ww]) for ww in range(w_block)] for hh in range(h_block)] for ii
                      in range(dataset.__len__())]
    if modality == 'raw2flow':
        foreground_set2 = [[[np.array(foreground_set2[ii][hh][ww]) for ww in range(w_block)] for hh in range(h_block)]
                           for ii in range(dataset.__len__())]

    foreground_bbox_set = [
        [[np.array(foreground_bbox_set[ii][hh][ww]) for ww in range(w_block)] for hh in range(h_block)] for ii in
        range(dataset.__len__())]
    if modality == 'raw2flow':
        np.save(os.path.join(data_root_dir, modality,
                             dataset_name + '_' + 'foreground_test_{}_{}_border_{}-raw.npy'.format(foreground_extraction_mode,
                                                                                                   context_frame_num, border_mode)),
                foreground_set)
        np.save(os.path.join(data_root_dir, modality,
                             dataset_name + '_' + 'foreground_test_{}_{}_border_{}-flow.npy'.format(foreground_extraction_mode, context_frame_num, border_mode)),
                foreground_set2)
    else:
        np.save(os.path.join(data_root_dir, modality,
                             dataset_name + '_' + 'foreground_test_{}_{}_border_{}.npy'.format(foreground_extraction_mode, context_frame_num, border_mode)),
                foreground_set)
    np.save(os.path.join(data_root_dir, modality,
                         dataset_name + '_' + 'foreground_bbox_test_{}.npy'.format(foreground_extraction_mode)),
            foreground_bbox_set)
    print('foreground for testing data saved!')
else:
    if dataset_name == 'ShanghaiTech':
        scene_idx = np.load(os.path.join(data_root_dir, modality, dataset_name + '_' + 'scene_idx.npy'))
    if modality == 'raw2flow':
        foreground_set = np.load(os.path.join(data_root_dir, modality,
                                              dataset_name + '_' + 'foreground_test_{}_{}_border_{}-raw.npy'.format(
                                                  foreground_extraction_mode, context_frame_num, border_mode)), allow_pickle=True)
        foreground_set2 = np.load(os.path.join(data_root_dir, modality,
                                               dataset_name + '_' + 'foreground_test_{}_{}_border_{}-flow.npy'.format(
                                                   foreground_extraction_mode, context_frame_num, border_mode)), allow_pickle=True)
    else:
        foreground_set = np.load(os.path.join(data_root_dir, modality,
                                              dataset_name + '_' + 'foreground_test_{}_{}_border_{}.npy'.format(
                                                  foreground_extraction_mode, context_frame_num, border_mode)), allow_pickle=True)

    foreground_bbox_set = np.load(os.path.join(data_root_dir, modality,
                                               dataset_name + '_' + 'foreground_bbox_test_{}.npy'.format(
                                                   foreground_extraction_mode)), allow_pickle=True)
    print('foreground for testing data loaded!')

#  /*------------------------------------------Abnormal event detection----------------------------------------------*/
results_dir = 'results'
scores_saved = args.scores_saved
big_number = 100000


time_start=time.time()

loss_func_perturb =  nn.MSELoss()

if scores_saved is False:
    if method == 'SelfComplete':
        h, w, _, sn = frame_size[dataset_name]
        if border_mode == 'predict':
            tot_frame_num = context_frame_num + 1
            tot_of_num = context_of_num + 1
        else:
            tot_frame_num = 2 * context_frame_num + 1
            tot_of_num = 2 * context_of_num + 1
        rawRange = 10
        if rawRange >= tot_frame_num:
            rawRange = None
        useFlow = args.use_flow
        padding = False

        assert modality == 'raw2flow'
        loss_func = nn.MSELoss(reduce=False)

        in_channels = 3
        pixel_result_dir = os.path.join(results_dir, dataset_name, 'score_mask_{}_head_{}_layer_{}_length_{}_pe_{}_epoch_{}_lambda_{}_{}_w_{}_{}_perturb_{}'.format(
                                           border_mode, num_heads, num_layers, context_frame_num, pe, epochs, args.lambda_raw, args.lambda_of, args.w_raw, args.w_of, args.epsilon) + '_' + 'pyname_{}.npy'.format(pyfile_name_score))
        os.makedirs(pixel_result_dir, exist_ok=True)

        model_weights = torch.load(os.path.join(data_root_dir, modality, dataset_name + '_' + 'model_{}_head_{}_layer_{}_length_{}_pe_{}_epoch_{}_lambda_{}_{}'.format(
             border_mode, num_heads, num_layers, context_frame_num, pe, epochs, args.lambda_raw, args.lambda_of) + '_' + 'pyname_{}.npy'.format(pyfile_name)))

        if dataset_name == 'ShanghaiTech':
            model_set = [[[[] for ww in range(len(model_weights[ss][hh]))] for hh in range(len(model_weights[ss]))]
                         for ss in range(len(model_weights))]
            for ss in range(len(model_weights)):
                for hh in range(len(model_weights[ss])):
                    for ww in range(len(model_weights[ss][hh])):
                        if len(model_weights[ss][hh][ww]) > 0:
                            cur_model = torch.nn.DataParallel(
                                ConvTransformer_recon_correct(
                                    tot_raw_num=tot_frame_num, nums_hidden=[32, 64, 128], num_layers=num_layers, 
                                    num_dec_frames=1, num_heads=num_heads, with_residual=True,
                                    with_pos=True, pos_kind=pe, mode=0, use_flow=args.use_flow)).cuda()
                            cur_model.load_state_dict(model_weights[ss][hh][ww][0])
                            model_set[ss][hh][ww].append(cur_model.eval())
            #  get training scores statistics
            raw_training_scores_set = torch.load(os.path.join(data_root_dir, modality,
                                                              dataset_name + '_' + 'raw_training_scores_border_{}_head_{}_layer_{}_length_{}_pe_{}_epoch_{}_lambda_{}_{}'.format(
                                                             border_mode, num_heads, num_layers, context_frame_num, pe,
                                                             epochs, args.lambda_raw, args.lambda_of)+ '_' + 'pyname_{}.npy'.format(pyfile_name)))
            of_training_scores_set = torch.load(os.path.join(data_root_dir, modality,
                                                             dataset_name + '_' + 'of_training_scores_border_{}_head_{}_layer_{}_length_{}_pe_{}_epoch_{}_lambda_{}_{}'.format(
                                                             border_mode, num_heads, num_layers, context_frame_num, pe,
                                                             epochs, args.lambda_raw, args.lambda_of) + '_' + 'pyname_{}.npy'.format(pyfile_name)))
            raw_stats_set = [[[(np.mean(raw_training_scores_set[ss][hh][ww]),
                                np.std(raw_training_scores_set[ss][hh][ww])) for ww in range(len(model_weights[hh]))]
                              for hh in range(len(model_weights))] for ss in range(len(model_weights))]
            if useFlow:
                of_stats_set = [[[(np.mean(of_training_scores_set[ss][hh][ww]),
                                   np.std(of_training_scores_set[ss][hh][ww])) for ww in range(len(model_weights[hh]))]
                                 for hh in range(len(model_weights))] for ss in range(len(model_weights))]
            del raw_training_scores_set, of_training_scores_set
        else:
            model_set = [[[] for ww in range(len(model_weights[hh]))] for hh in range(len(model_weights))]


            for hh in range(len(model_weights)):
                for ww in range(len(model_weights[hh])):
                    if len(model_weights[hh][ww]) > 0:
                        cur_model = torch.nn.DataParallel(
                            ConvTransformer_recon_correct(
                                tot_raw_num=tot_frame_num, nums_hidden=[32, 64, 128], num_layers=num_layers,
                                num_dec_frames=1, num_heads=num_heads, with_residual=True,
                                with_pos=True, pos_kind=pe, mode=0, use_flow=args.use_flow)).cuda()
                        print(model_weights[hh][ww][0].keys())
                        cur_model.load_state_dict(model_weights[hh][ww][0])

                        model_set[hh][ww].append(cur_model.eval())
            #  get training scores statistics
            raw_training_scores_set = torch.load(os.path.join(data_root_dir, modality,
                                                              dataset_name + '_' + 'raw_training_scores_border_{}_head_{}_layer_{}_length_{}_pe_{}_epoch_{}_lambda_{}_{}'.format(
                                                                   border_mode, num_heads, num_layers, context_frame_num, pe, epochs, args.lambda_raw, args.lambda_of) + '_' + 'pyname_{}.npy'.format(pyfile_name)))
            of_training_scores_set = torch.load(os.path.join(data_root_dir, modality,
                                                             dataset_name + '_' + 'of_training_scores_border_{}_head_{}_layer_{}_length_{}_pe_{}_epoch_{}_lambda_{}_{}'.format( border_mode, num_heads, num_layers, context_frame_num, pe, epochs, args.lambda_raw, args.lambda_of) + '_' + 'pyname_{}.npy'.format(pyfile_name)))

            # mean and std of training scores
            raw_stats_set = [
                [(np.mean(raw_training_scores_set[hh][ww]), np.std(raw_training_scores_set[hh][ww])) for ww in
                 range(len(model_weights[hh]))] for hh in range(len(model_weights))]
            if useFlow:
                of_stats_set = [
                    [(np.mean(of_training_scores_set[hh][ww]), np.std(of_training_scores_set[hh][ww])) for ww in
                     range(len(model_weights[hh]))] for hh in range(len(model_weights))]

            del raw_training_scores_set, of_training_scores_set

        # Get scores
        for frame_idx in range(len(foreground_set)):
            print('Calculating scores for {}-th frame'.format(frame_idx))
            cur_data_set = foreground_set[frame_idx]
            cur_data_set2 = foreground_set2[frame_idx]
            cur_bboxes = foreground_bbox_set[frame_idx]
            cur_pixel_results = -1 * np.ones(shape=(h, w)) * big_number
            for h_idx in range(len(cur_data_set)):
                for w_idx in range(len(cur_data_set[h_idx])):
                    if len(cur_data_set[h_idx][w_idx]) > 0:

                        if dataset_name == 'ShanghaiTech':
                            if len(model_set[scene_idx[frame_idx] - 1][h_idx][w_idx]) > 0:
                                # print(scene_idx[frame_idx])
                                cur_model = model_set[scene_idx[frame_idx] - 1][h_idx][w_idx][0]
                                cur_dataset = cube_to_train_dataset(cur_data_set[h_idx][w_idx],
                                                                    target=cur_data_set2[h_idx][w_idx])
                                cur_dataloader = DataLoader(dataset=cur_dataset,
                                                            batch_size=cur_data_set[h_idx][w_idx].shape[0],
                                                            shuffle=False)
                                for idx, (inputs, of_targets_all, _) in enumerate(cur_dataloader):
                                    inputs = inputs.cuda().type(torch.cuda.FloatTensor)
                                    inputs =  torch.autograd.Variable(inputs, requires_grad= True)
                                    of_targets_all = of_targets_all.cuda().type(torch.cuda.FloatTensor)
                                    of_outputs, raw_outputs, of_targets, raw_targets = cur_model(inputs, of_targets_all)

                                    loss_raw = loss_func_perturb(raw_targets, raw_outputs)
                                    if useFlow:
                                        loss_of = loss_func_perturb(of_targets.detach(), of_outputs)
                            
                                    if useFlow:
                                        loss = loss_raw + loss_of
                                    else:
                                        loss = loss_raw
                                    loss.backward()
                                
                                    gradient = inputs.grad.data
                                    sign_gradient = torch.sign(gradient)
                                    middle_start_indice = 3*context_frame_num

                                    inputs.requires_grad = False
                                    inputs = torch.add(inputs.data, -args.epsilon, sign_gradient)

                                    # end of perturb

                                    inputs =  torch.autograd.Variable(inputs)
                                    of_outputs, raw_outputs, of_targets, raw_targets = cur_model(inputs, of_targets_all)

                                    # # visualization
                                    # for i in range(raw_targets.size(0)):
                                    #     visualize_recon(
                                    #     batch_1=img_batch_tensor2numpy(raw_targets.cpu().detach()[i]),
                                    #     batch_2=img_batch_tensor2numpy(raw_outputs.cpu().detach()[i]),
                                    #     frame_idx=frame_idx, obj_id = i, dataset_name = dataset_name, save_dir=visual_save_dir)
                                    #     visualize_recon(
                                    #     batch_1=img_batch_tensor2numpy(of_targets.cpu().detach()[i]),
                                    #     batch_2=img_batch_tensor2numpy(of_outputs.cpu().detach()[i]),
                                    #     frame_idx=frame_idx, obj_id = i, dataset_name = dataset_name, save_dir=visual_save_dir)


                                    if useFlow:
                                        of_scores = loss_func(of_targets, of_outputs).cpu().data.numpy()
                                        of_scores = np.sum(np.sum(np.sum(np.sum(of_scores, axis=4), axis=3), axis=2), axis=1)
                                        # print(of_scores)# mse

                                    raw_scores = loss_func(raw_targets, raw_outputs).cpu().data.numpy()
                                    raw_scores = np.sum(np.sum(np.sum(np.sum(raw_scores, axis=4), axis=3), axis=2), axis=1)  # mse
                                    # print(raw_scores)

                                    raw_scores = (raw_scores - raw_stats_set[scene_idx[frame_idx] - 1][h_idx][w_idx][
                                        0]) / raw_stats_set[scene_idx[frame_idx] - 1][h_idx][w_idx][1]
                                    # print(raw_scores)
                                    if useFlow:
                                        of_scores = (of_scores - of_stats_set[scene_idx[frame_idx] - 1][h_idx][w_idx][
                                            0]) / of_stats_set[scene_idx[frame_idx] - 1][h_idx][w_idx][1]
                                        # print(of_scores)
                                    if useFlow:
                                        scores = args.w_raw * raw_scores + args.w_of* of_scores
                                        # print(scores)
                                    else:
                                        scores = args.w_raw * raw_scores

                            else:
                                scores = np.ones(cur_data_set[h_idx][w_idx].shape[0], ) * big_number


                        else:
                            if len(model_set[h_idx][w_idx]) > 0:
                                cur_model = model_set[h_idx][w_idx][0]
                                cur_dataset = cube_to_train_dataset(cur_data_set[h_idx][w_idx],
                                                                    target=cur_data_set2[h_idx][w_idx])
                                cur_dataloader = DataLoader(dataset=cur_dataset,
                                                            batch_size=cur_data_set[h_idx][w_idx].shape[0],
                                                            shuffle=False)

                                for idx, (inputs, of_targets_all, _) in enumerate(cur_dataloader):
                            
                                    inputs = inputs.cuda().type(torch.cuda.FloatTensor)
      

                                    inputs =  torch.autograd.Variable(inputs, requires_grad= True)

            
                                    of_targets_all = of_targets_all.cuda().type(torch.cuda.FloatTensor)
                                    of_outputs, raw_outputs, of_targets, raw_targets = cur_model(inputs, of_targets_all)

                                    loss_raw = loss_func_perturb(raw_targets, raw_outputs)
                                    if useFlow:
                                        loss_of = loss_func_perturb(of_targets.detach(), of_outputs)
                            
                                    if useFlow:
                                        loss = loss_raw + loss_of
                                    else:
                                        loss = loss_raw
                                    loss.backward()

                                    gradient = inputs.grad.data
                                    sign_gradient = torch.sign(gradient)
                                    middle_start_indice = 3*context_frame_num

                                    inputs.requires_grad = False
                                    inputs = torch.add(inputs.data, -args.epsilon, sign_gradient)


                                    # end of perturb

                                    inputs =  torch.autograd.Variable(inputs)
                                    of_outputs, raw_outputs, of_targets, raw_targets = cur_model(inputs, of_targets_all)






                                    # # visualization
                                    # for i in range(raw_targets.size(0)):
                                    #     visualize_recon(
                                    #     batch_1=img_batch_tensor2numpy(raw_targets.cpu().detach()[i]),
                                    #     batch_2=img_batch_tensor2numpy(raw_outputs.cpu().detach()[i]),
                                    #     frame_idx=frame_idx, obj_id = i, dataset_name = dataset_name, save_dir=visual_save_dir)
                                    #     visualize_recon(
                                    #     batch_1=img_batch_tensor2numpy(of_targets.cpu().detach()[i]),
                                    #     batch_2=img_batch_tensor2numpy(of_outputs.cpu().detach()[i]),
                                    #     frame_idx=frame_idx, obj_id = i, dataset_name = dataset_name, save_dir=visual_save_dir)
                        



                                    # mse
                                    if useFlow:
                                        of_scores = loss_func(of_targets, of_outputs).cpu().data.numpy()
                                        # of_scores = np.sum(of_scores, axis=(4, 3, 2))  # bl
                                        #
                                        # for l in range(of_scores.shape[1]):
                                        #     of_scores[:, l] = of_scores[:, l] * (abs(l - context_frame_num) + 1)
                                        # of_scores = np.sum(of_scores, axis=1)
                                        of_scores = np.sum(np.sum(np.sum(np.sum(of_scores, axis=4), axis=3), axis=2), axis=1)

                                    raw_scores = loss_func(raw_targets, raw_outputs).cpu().data.numpy()
                                    raw_scores = np.sum(np.sum(np.sum(np.sum(raw_scores, axis=4), axis=3), axis=2), axis=1)
                                    # raw_scores = np.sum(raw_scores, axis=(4, 3, 2))  # bl
                                    #
                                    # for l in range(raw_scores.shape[1]):
                                    #     raw_scores[:, l] = raw_scores[:, l] * (abs(l - context_frame_num)+1)
                                    # raw_scores = np.sum(raw_scores, axis=1)

                                    # normalize scores using training scores
                                    raw_scores = (raw_scores - raw_stats_set[h_idx][w_idx][0]) / \
                                                 raw_stats_set[h_idx][w_idx][1]
                                    if useFlow:
                                        of_scores = (of_scores - of_stats_set[h_idx][w_idx][0]) / \
                                                    of_stats_set[h_idx][w_idx][1]

                                    if useFlow:
                                        scores = args.w_raw * raw_scores + args.w_of * of_scores
                                    else:
                                        scores = args.w_raw * raw_scores
                                    # print(scores.shape)
                            else:
                                scores = np.ones(cur_data_set[h_idx][w_idx].shape[0], ) * big_number

                        for m in range(scores.shape[0]):
                            cur_score_mask = -1 * np.ones(shape=(h, w)) * big_number
                            cur_score = scores[m]
                            bbox = cur_bboxes[h_idx][w_idx][m]
                            x_min, x_max = np.int(np.ceil(bbox[0])), np.int(np.ceil(bbox[2]))
                            y_min, y_max = np.int(np.ceil(bbox[1])), np.int(np.ceil(bbox[3]))
                            cur_score_mask[y_min:y_max, x_min:x_max] = cur_score
                            cur_pixel_results = np.max(
                                np.concatenate([cur_pixel_results[:, :, np.newaxis], cur_score_mask[:, :, np.newaxis]],
                                               axis=2), axis=2)
            torch.save(cur_pixel_results, os.path.join(pixel_result_dir, '{}'.format(frame_idx)))
    else:
        raise NotImplementedError

#  /*------------------------------------------Evaluation----------------------------------------------*/
criterion = 'frame'
batch_size = 1
# set dataset for evaluation
dataset = unified_dataset_interface(dataset_name=dataset_name, dir=os.path.join(raw_dataset_dir, dataset_name),
                                    context_frame_num=0, mode=mode, border_mode='hard')
dataset_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                            collate_fn=bbox_collate(mode).collate)

print('Evaluating {} by {}-criterion:'.format(dataset_name, criterion))
if criterion == 'frame':
    if dataset_name == 'ShanghaiTech':
        all_frame_scores = [[] for si in set(dataset.scene_idx)]
        all_targets = [[] for si in set(dataset.scene_idx)]
        for idx, (_, target) in enumerate(dataset_loader):
            print('Processing {}-th frame'.format(idx))
            cur_pixel_results = torch.load(os.path.join(results_dir, dataset_name, 'score_mask_{}_head_{}_layer_{}_length_{}_pe_{}_epoch_{}_lambda_{}_{}_w_{}_{}_perturb_{}'.format(
                                           border_mode, num_heads, num_layers, context_frame_num, pe, epochs, args.lambda_raw, args.lambda_of, args.w_raw, args.w_of, args.epsilon) + '_' + 'pyname_{}.npy'.format(pyfile_name_score), '{}'.format(idx) ))
            all_frame_scores[scene_idx[idx] - 1].append(cur_pixel_results.max())
            all_targets[scene_idx[idx] - 1].append(target[0].numpy().max())
        all_frame_scores = [np.array(all_frame_scores[si]) for si in range(dataset.scene_num)]
        all_targets = [np.array(all_targets[si]) for si in range(dataset.scene_num)]
        all_targets = [all_targets[si] > 0 for si in range(dataset.scene_num)]
        print(dataset.scene_num)
        print(all_frame_scores)
        print(all_targets)
        results = [save_roc_pr_curve_data(all_frame_scores[si], all_targets[si], os.path.join(results_dir, dataset_name,
                                                                                              '{}_{}_{}_frame_results_scene_{}.npz'.format(
                                                                                                  modality,
                                                                                                  foreground_extraction_mode,
                                                                                                  method, si + 1))) for
                   si in range(dataset.scene_num)]
        results = np.array(results).mean()
        print('Average frame-level AUC is {}'.format(results))
        print(np.max(all_frame_scores))
        print(np.min(all_frame_scores))
    else:
        all_frame_scores = list()
        all_targets = list()
        for idx, (_, target) in enumerate(dataset_loader):
            print('Processing {}-th frame'.format(idx))
            cur_pixel_results = torch.load(os.path.join(results_dir, dataset_name, 'score_mask_{}_head_{}_layer_{}_length_{}_pe_{}_epoch_{}_lambda_{}_{}_w_{}_{}_perturb_{}'.format(
                                           border_mode, num_heads, num_layers, context_frame_num, pe, epochs, args.lambda_raw, args.lambda_of, args.w_raw, args.w_of, args.epsilon) + '_' + 'pyname_{}.npy'.format(pyfile_name_score), '{}'.format(idx)))
            all_frame_scores.append(cur_pixel_results.max())
            all_targets.append(target[0].numpy().max())

        time_end = time.time()
        print('time cost', time_end - time_start, 's')
        all_frame_scores = np.array(all_frame_scores)
        all_targets = np.array(all_targets)
        all_targets = all_targets > 0
        results_path = os.path.join(results_dir, dataset_name,
                                    '{}_{}_{}_frame_results.npz'.format(modality, foreground_extraction_mode, method))
        print('Results written to {}:'.format(results_path))
        np.save('output_scores_{}_pyname_{}'.format(dataset_name, pyfile_name_score), all_frame_scores)
        np.save('labels_{}'.format(dataset_name), all_targets)
        print(all_frame_scores)
        print(all_targets)
        auc = save_roc_pr_curve_data(all_frame_scores, all_targets, results_path,verbose=True)
        print(auc)
elif criterion == 'pixel':
    if dataset_name != 'ShanghaiTech':
        all_pixel_scores = list()
        all_targets = list()
        thr = 0.4
        for idx, (_, target) in enumerate(dataset_loader):
            print('Processing {}-th frame'.format(idx))
            cur_pixel_results = torch.load(os.path.join(results_dir, dataset_name, 'score_mask', '{}'.format(idx)))
            target_mask = target[0].numpy()
            all_targets.append(target[0].numpy().max())
            if all_targets[-1] > 0:
                cur_effective_scores = cur_pixel_results[target_mask > 0]
                sorted_score = np.sort(cur_effective_scores)
                cut_off_idx = np.int(np.round((1 - thr) * cur_effective_scores.shape[0]))
                cut_off_score = cur_effective_scores[cut_off_idx]
            else:
                cut_off_score = cur_pixel_results.max()
            all_pixel_scores.append(cut_off_score)
        all_frame_scores = np.array(all_pixel_scores)
        all_targets = np.array(all_targets)
        all_targets = all_targets > 0
        results_path = os.path.join(results_dir, dataset_name,
                                    '{}_{}_{}_pixel_results.npz'.format(modality, foreground_extraction_mode, method))
        print('Results written to {}:'.format(results_path))
        results = save_roc_pr_curve_data(all_frame_scores, all_targets, results_path)
    else:
        raise NotImplementedError
else:
    raise NotImplementedError
