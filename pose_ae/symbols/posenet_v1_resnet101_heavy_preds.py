import mxnet as mx
import numpy as np
from common.lib.utils.symbol import Symbol
from common.backbone import resnet_v1
from common.gpu_metric import *
from common.operator_py.a_loss import *
from common.operator_py.d_loss import *
from common.operator_py.monitor_op import *


class posenet_v1_resnet101_heavy_preds(Symbol):
    def __init__(self, FP16=False):
        """
        Use __init__ to define parameter network needs
        """
        # FP16 is not used for now
        self.FP16 = FP16
        self.init_pre_list = []
        self.init_hourglass_list = []

    def get_symbol(self, cfg, is_train=True):
        # config alias for convenient
        num_parts = cfg.dataset.NUM_PARTS
        max_persons = cfg.dataset.MAX_PERSONS
        feat_w, feat_h = cfg.pose_aug.SCALES_OUT[0]

        # input init
        if is_train:
            data = mx.sym.Variable(name="data")  # img, [N, 3, H ,W]
            heatmaps = mx.sym.Variable(name="heatmaps")  # heatmaps of parts, [N, num_parts, H/4, W/4], REMARK 1/4 scale
            masks = mx.sym.Variable(name="masks")  # mask of crowds in coco, [N, H/4, W/4], REMARK 1/4 scale
            keypoints = mx.sym.Variable(name='keypoints')  # coordinates of keypoints, [N, max_persons, num_parts, 2], REMARK 1/4 scale
            bn_use_global_stats = False
        else:
            data = mx.sym.Variable(name="data")  # img, [N, 3, H ,W]
            bn_use_global_stats = True

        _, _, _, _, c5 = resnet_v1.get_resnet_backbone(data=data, num_layers=101,
                                                       use_dilation_on_c5=False,
                                                       use_dconv=False, dconv_lr_mult=0.001, dconv_group=1, dconv_start_channel=512,
                                                       bn_mom=0.9)
        # simple baseline's deconv net
        data = c5
        for _stage in range(3):
            prefix = 'simple_baseline_stage{}'.format(_stage)
            _stage_scale = 2 ** (2 - _stage)
            data = mx.sym.Deconvolution(data=data, num_filter=256, kernel=(4, 4), stride=(2, 2),
                                        no_bias=False, target_shape=(feat_h / _stage_scale, feat_w / _stage_scale),
                                        name=prefix + '_deconv')
            # data = mx.sym.BatchNorm(data=data, use_global_stats=bn_use_global_stats, fix_gamma=False, eps=2e-5,
            #                         momentum=0.9, name=prefix + '_bn')
            data = mx.sym.Activation(data=data, act_type='relu', name=prefix + '_relu')

        d_preds = mx.sym.Convolution(data=data, num_filter=num_parts, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                     no_bias=False, name='simple_baseline_d_preds_conv1')  # shape, [N, num_parts, H, W]
        d_preds = mx.sym.Activation(data=d_preds, act_type='relu', name='simple_baseline_d_preds_relu1')
        d_preds = mx.sym.Convolution(data=d_preds, num_filter=num_parts, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                     no_bias=False, name='simple_baseline_d_preds_conv2')  # shape, [N, num_parts, H, W]
        d_preds = mx.sym.Activation(data=d_preds, act_type='relu', name='simple_baseline_d_preds_relu2')
        d_preds = mx.sym.Convolution(data=d_preds, num_filter=num_parts, kernel=(1, 1), stride=(1, 1),
                                     no_bias=False, name='simple_baseline_d_preds3')  # shape, [N, num_parts, H, W]

        a_preds = mx.sym.Convolution(data=data, num_filter=num_parts, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                     no_bias=False, name='simple_baseline_a_preds_conv1')  # shape, [N, num_parts, H, W]
        a_preds = mx.sym.Activation(data=a_preds, act_type='relu', name='simple_baseline_a_preds_relu1')
        a_preds = mx.sym.Convolution(data=a_preds, num_filter=num_parts, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                     no_bias=False, name='simple_baseline_a_preds_conv2')  # shape, [N, num_parts, H, W]
        a_preds = mx.sym.Activation(data=a_preds, act_type='relu', name='simple_baseline_a_preds_relu2')
        a_preds = mx.sym.Convolution(data=a_preds, num_filter=num_parts, kernel=(1, 1), stride=(1, 1),
                                     no_bias=False, name='simple_baseline_a_preds_conv3')  # shape, [N, num_parts, H, W]

        # calc_loss
        if is_train:
            # calc detection loss
            tmp_d_loss = mx.symbol.square(data=(d_preds - heatmaps))
            masks_expand = mx.symbol.expand_dims(masks, axis=1)
            tmp_d_loss = mx.symbol.broadcast_mul(lhs=tmp_d_loss, rhs=masks_expand)
            tmp_d_loss = mx.symbol.mean(data=tmp_d_loss, axis=(1, 2, 3))  # shape, [N]

            d_loss = mx.symbol.mean(data=tmp_d_loss, axis=0)  # shape, [1]

            # prepare keypoints
            keypoints_idx_list = []
            keypoints_mask_list = []
            keypoints_valid_list = []
            keypoints_valid_list2 = []
            for i in range(max_persons):
                tmp_keypoints = mx.symbol.squeeze(mx.sym.slice_axis(data=keypoints, axis=1, begin=i, end=i + 1), axis=(1))  # shape, [N, num_parts, 2]
                tmp_keypoints_idx = mx.symbol.squeeze(mx.sym.slice_axis(data=tmp_keypoints, axis=2, begin=0, end=1), axis=(2))  # shape, [N, num_parts]
                tmp_keypoints_mask = mx.symbol.squeeze(mx.sym.slice_axis(data=tmp_keypoints, axis=2, begin=1, end=2), axis=(2))  # shape, [N, num_parts]
                tmp_keypoints_valid = tmp_keypoints_mask.sum(axis=1) > 0  # shape, [N]
                keypoints_idx_list.append(tmp_keypoints_idx)
                keypoints_mask_list.append(tmp_keypoints_mask)
                keypoints_valid_list.append(tmp_keypoints_valid)
            tmp_a_loss_inside_count = mx.sym.stack(*keypoints_valid_list, axis=1)  # shape [N, max_persons]
            tmp_a_loss_inside_scale = mx.symbol.broadcast_div(tmp_a_loss_inside_count, tmp_a_loss_inside_count.sum(axis=1, keepdims=True) + (tmp_a_loss_inside_count.sum(axis=1, keepdims=True) == 0))  # shape, [N, max_persons]

            for j in range(max_persons):
                for k in range(j + 1, max_persons):
                    keypoints_valid_list2.append(keypoints_valid_list[j] * keypoints_valid_list[k])
            tmp_a_loss_outside_count = mx.sym.stack(*keypoints_valid_list2, axis=1)  # shape, [N, max_persons * (max_persons - 1) / 2]
            tmp_a_loss_outside_scale = mx.symbol.broadcast_div(tmp_a_loss_outside_count, tmp_a_loss_outside_count.sum(axis=1, keepdims=True) + (tmp_a_loss_outside_count.sum(axis=1, keepdims=True) == 0))  # shape, [N, max_persons * (max_persons - 1) / 2]

            # calc association loss

            # flatten
            tmp_a_pred = mx.sym.Reshape(data=a_preds, shape=(0, 0, -3), name='association_reshape{}'.format(i + 1))  # shape, [N, num_parts, W/4 * H/4]
            person_mean_list = []

            # calc association loss inside one person
            tmp_a_loss_inside_list = []
            for j in range(max_persons):
                # pick feat for person-j
                tmp_person_feat = mx.sym.pick(tmp_a_pred, keypoints_idx_list[j])  # shape, [N, num_parts]
                # calc person-j's mean feat
                tmp_persin_feat_scale = mx.symbol.broadcast_div(keypoints_mask_list[j], keypoints_mask_list[j].sum(axis=1, keepdims=True) + (keypoints_mask_list[j].sum(axis=1, keepdims=True) == 0))  # shape, [N, num_parts]
                tmp_person_feat_mean = mx.symbol.sum(tmp_person_feat * tmp_persin_feat_scale, axis=1)  # shape, [N]

                person_mean_list.append(tmp_person_feat_mean)
                tmp_person_feat_mean_expand = mx.symbol.expand_dims(tmp_person_feat_mean, axis=1)  # shape, [N, 1]

                # calc a_loss_inside
                tmp_a_loss_inside = mx.symbol.square(data=mx.symbol.broadcast_sub(lhs=tmp_person_feat, rhs=tmp_person_feat_mean_expand))  # shape [N, num_parts]
                tmp_a_loss_inside = mx.symbol.sum(tmp_a_loss_inside * tmp_persin_feat_scale, axis=1)  # shape, [N]
                tmp_a_loss_inside_list.append(tmp_a_loss_inside)
            tmp_a_loss_inside = mx.sym.stack(*tmp_a_loss_inside_list, axis=1)  # shape, [N, max_persons]
            tmp_a_loss_inside = mx.symbol.sum(tmp_a_loss_inside * tmp_a_loss_inside_scale, axis=1)  # shape, [N]

            # calc association loss inside between persons
            tmp_a_loss_outside_list = []
            tmp_ids = 0
            for j in range(max_persons):
                for k in range(j + 1, max_persons):
                    tmp_a_loss_outside = mx.symbol.exp(data=-mx.symbol.square(
                        data=(person_mean_list[j] - person_mean_list[k])))  # -1/2sigma^2 is -1 here, shape, [N]
                    tmp_a_loss_outside = tmp_a_loss_outside * keypoints_valid_list2[tmp_ids]
                    tmp_a_loss_outside_list.append(tmp_a_loss_outside)
                    tmp_ids += 1
            tmp_a_loss_outside = mx.sym.stack(*tmp_a_loss_outside_list, axis=1)  # shape, [N, max_persons * (max_persons - 1) / 2]
            tmp_a_loss_outside = mx.symbol.sum(tmp_a_loss_outside * tmp_a_loss_outside_scale, axis=1)  # shape, [N]

            # stack all stage
            tmp_a_loss_inside = mx.symbol.mean(data=tmp_a_loss_inside, axis=0)  # shape, [1]
            tmp_a_loss_outside = 0.5 * mx.symbol.mean(data=tmp_a_loss_outside, axis=0)  # shape, [1]

            # mask Loss
            d_loss = mx.sym.MakeLoss(name='detection_loss', data=d_loss, grad_scale=cfg.TRAIN.d_loss)
            a_loss_inside = mx.sym.MakeLoss(name='association_loss_inside', data=tmp_a_loss_inside, grad_scale=cfg.TRAIN.a_loss_in)
            a_loss_outside = mx.sym.MakeLoss(name='association_loss_outside', data=tmp_a_loss_outside, grad_scale=cfg.TRAIN.a_loss_out)

            output_list = [d_loss, a_loss_inside, a_loss_outside]

            # get gpu metric
            if cfg.TRAIN.GPU_METRIC:
                output_list.extend(get_detection_loss(tmp_d_loss))
                output_list.extend(get_association_loss_inside(tmp_a_loss_inside))
                output_list.extend(get_association_loss_outside(tmp_a_loss_outside))
                output_list.extend(get_det_max(d_preds))
                output_list.extend(get_tag_mean(a_preds))
                output_list.extend(get_tag_var(a_preds))
            else:
                raise ValueError('No CPU metric is supported now!')

            group = mx.sym.Group(output_list)
        else:
            group = mx.sym.Group([d_preds, a_preds])

        self.sym = group
        return group

    def get_pred_names(self, is_train, gpu_metric=False):
        if is_train:
            pred_names = ['d_loss', 'a_loss_inside', 'a_loss_outside']
            if gpu_metric:
                pred_names.extend([
                    'D_Loss',
                    'A_Loss_Inside',
                    'A_Loss_Outside',
                    'Det_Max',
                    'Tag_Mean',
                    'Tag_Var'])
            return pred_names
        else:
            pred_names = ['d_loss', 'a_loss_inside', 'a_loss_outside']

    def get_label_names(self):
        return ['d_preds, a_preds']

    def init_weight_simple_baseline(self, cfg, arg_params, aux_params):

        for _stage in range(3):
            prefix = 'simple_baseline_stage{}'.format(_stage)
            # pytorch's kaiming_uniform_
            weight_shape = self.arg_shape_dict[prefix + '_deconv_weight']
            fan_in = float(weight_shape[1]) * weight_shape[2] * weight_shape[3]
            bound = np.sqrt(6 / ((1 + 5) * fan_in))
            arg_params[prefix + '_deconv_weight'] = mx.random.uniform(-bound, bound, shape=weight_shape)
            arg_params[prefix + '_deconv_bias'] = mx.random.uniform(-bound, bound, shape=self.arg_shape_dict[prefix + '_deconv_bias'])

            # arg_params[prefix + '_bn_gamma'] = mx.random.uniform(0, 1, shape=self.arg_shape_dict[prefix + '_bn_gamma'])
            # arg_params[prefix + '_bn_beta'] = mx.nd.zeros(shape=self.arg_shape_dict[prefix + '_bn_beta'])
            # aux_params[prefix + '_bn_moving_mean'] = mx.nd.zeros(shape=self.aux_shape_dict[prefix + '_bn_moving_mean'])
            # aux_params[prefix + '_bn_moving_var'] = mx.nd.ones(shape=self.aux_shape_dict[prefix + '_bn_moving_var'])

        # pytorch's kaiming_uniform_
        weight_shape = self.arg_shape_dict['simple_baseline_d_preds_conv1_weight']
        fan_in = float(weight_shape[1]) * weight_shape[2] * weight_shape[3]
        bound = np.sqrt(6 / ((1 + 5) * fan_in))
        arg_params['simple_baseline_d_preds_conv1_weight'] = mx.random.uniform(-bound, bound, shape=weight_shape)
        arg_params['simple_baseline_d_preds_conv1_bias'] = mx.random.uniform(-bound, bound, shape=self.arg_shape_dict['simple_baseline_d_preds_conv1_bias'])
        arg_params['simple_baseline_a_preds_conv1_weight'] = mx.random.uniform(-bound, bound, shape=self.arg_shape_dict['simple_baseline_a_preds_conv1_weight'])
        arg_params['simple_baseline_a_preds_conv1_bias'] = mx.random.uniform(-bound, bound, shape=self.arg_shape_dict['simple_baseline_a_preds_conv1_bias'])

        weight_shape = self.arg_shape_dict['simple_baseline_d_preds_conv2_weight']
        fan_in = float(weight_shape[1]) * weight_shape[2] * weight_shape[3]
        bound = np.sqrt(6 / ((1 + 5) * fan_in))
        arg_params['simple_baseline_d_preds_conv2_weight'] = mx.random.uniform(-bound, bound, shape=weight_shape)
        arg_params['simple_baseline_d_preds_conv2_bias'] = mx.random.uniform(-bound, bound, shape=self.arg_shape_dict['simple_baseline_d_preds_conv2_bias'])
        arg_params['simple_baseline_a_preds_conv2_weight'] = mx.random.uniform(-bound, bound, shape=self.arg_shape_dict['simple_baseline_a_preds_conv2_weight'])
        arg_params['simple_baseline_a_preds_conv2_bias'] = mx.random.uniform(-bound, bound, shape=self.arg_shape_dict['simple_baseline_a_preds_conv2_bias'])

        weight_shape = self.arg_shape_dict['simple_baseline_d_preds_conv3_weight']
        fan_in = float(weight_shape[1]) * weight_shape[2] * weight_shape[3]
        bound = np.sqrt(6 / ((1 + 5) * fan_in))
        arg_params['simple_baseline_d_preds_conv3_weight'] = mx.random.uniform(-bound, bound, shape=weight_shape)
        arg_params['simple_baseline_d_preds_conv3_bias'] = mx.random.uniform(-bound, bound, shape=self.arg_shape_dict['simple_baseline_d_preds_conv3_bias'])
        arg_params['simple_baseline_a_preds_conv3_weight'] = mx.random.uniform(-bound, bound, shape=self.arg_shape_dict['simple_baseline_a_preds_conv3_weight'])
        arg_params['simple_baseline_a_preds_conv3_bias'] = mx.random.uniform(-bound, bound, shape=self.arg_shape_dict['simple_baseline_a_preds_conv3_bias'])

        # # a_preds branch's init
        # arg_params['simple_baseline_a_preds_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['simple_baseline_a_preds_weight'])
        # arg_params['simple_baseline_a_preds_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['simple_baseline_a_preds_bias'])

        '''
        # ones/zero for debug
        arg_params['{}_weight'.format(ele)] = 1 / 2000.0 * mx.nd.ones(shape=self.arg_shape_dict['{}_weight'.format(ele)])
        arg_params['{}_bias'.format(ele)] = mx.nd.zeros(shape=self.arg_shape_dict['{}_bias'.format(ele)])

        # pytorch's xavier_
        weight_shape = self.arg_shape_dict['{}_weight'.format(ele)]
        std = np.sqrt(2 / (float(weight_shape[0] + weight_shape[1]) * weight_shape[2] * weight_shape[3]))
        arg_params['{}_weight'.format(ele)] = mx.random.normal(0, std, shape=weight_shape)
        arg_params['{}_bias'.format(ele)] = mx.nd.zeros(shape=self.arg_shape_dict['{}_bias'.format(ele)])
        '''

    def init_weight(self, cfg, arg_params, aux_params):
        self.init_weight_simple_baseline(cfg, arg_params, aux_params)
