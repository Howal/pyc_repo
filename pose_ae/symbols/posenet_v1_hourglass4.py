import mxnet as mx
import numpy as np
from common.lib.utils.symbol import Symbol
from common.backbone.hourglass_v1 import get_stacked_hourglass, conv_sym_wrapper, CurrentBN
from common.gpu_metric import *
from common.operator_py.a_loss import *
from common.operator_py.d_loss import *
from common.operator_py.monitor_op import *


class posenet_v1_hourglass4(Symbol):
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
        in_dim = 256
        increase_dim = 128
        out_dim = 68
        num_stack = 4

        num_parts = cfg.dataset.NUM_PARTS
        max_persons = cfg.dataset.MAX_PERSONS

        # input init
        if is_train:
            data = mx.sym.Variable(name="data")  # img, [N, 3, H ,W]
            heatmaps = mx.sym.Variable(name="heatmaps")  # heatmaps of parts, [N, num_parts, H/4, W/4], REMARK 1/4 scale
            masks = mx.sym.Variable(name="masks")  # mask of crowds in coco, [N, H/4, W/4], REMARK 1/4 scale
            keypoints = mx.sym.Variable(name='keypoints')  # coordinates of keypoints, [N, max_persons, num_parts, 2], REMARK 1/4 scale
            # prepare BN func, this one can be easily replaced
            bn = CurrentBN(cfg.network.use_bn_type, 0.9)
        else:
            data = mx.sym.Variable(name="data")  # img, [N, 3, H ,W]
            # prepare BN func, this one can be easily replaced
            bn = CurrentBN(cfg.network.use_bn_type, 0.9, use_global_stats=False)

        # pre
        init_pre_list = []
        data = conv_sym_wrapper(data=data, prefix="pre1", num_filter=64, kernel=(7, 7),
                                stride=(2, 2), pad=(3, 3), bn=bn, record=init_pre_list)
        data = conv_sym_wrapper(data=data, prefix="pre2", num_filter=128, kernel=(3, 3),
                                stride=(1, 1), pad=(1, 1), bn=bn, record=init_pre_list)
        data = mx.symbol.Pooling(data=data, kernel=(2, 2), stride=(2, 2), pad=(0, 0), pool_type='max')
        data = conv_sym_wrapper(data=data, prefix="pre3", num_filter=128, kernel=(3, 3),
                                stride=(1, 1), pad=(1, 1), bn=bn, record=init_pre_list)
        data = conv_sym_wrapper(data=data, prefix="pre4", num_filter=in_dim, kernel=(3, 3),
                                stride=(1, 1), pad=(1, 1), bn=bn, record=init_pre_list)
        self.init_pre_list = init_pre_list

        # hourglass
        # preds->shape [N, num_stack, C=out_dim, H, W]
        init_hourglass_list = []
        preds = get_stacked_hourglass(data=data, num_stack=num_stack, in_dim=in_dim, out_dim=out_dim,
                                      increase_dim=increase_dim, bn=bn, record=init_hourglass_list)
        self.init_hourglass_list = init_hourglass_list

        # calc_loss
        if is_train:
            # slice into two parts
            d_preds = mx.sym.slice_axis(data=preds, axis=2, begin=0, end=num_parts)  # shape, [N, num_stack, num_parts, H, W]
            a_preds = mx.sym.slice_axis(data=preds, axis=2, begin=num_parts, end=2*num_parts)  # shape, [N, num_stack, num_parts, H, W]

            # a_preds = mx.symbol.Custom(op_type='monitor', input=a_preds, nickname='a_preds')
            # keypoints = mx.symbol.Custom(op_type='monitor', input=keypoints, nickname='keypoints')

            # calc detection loss
            d_loss = []
            for i in range(num_stack):
                tmp_d_pred = mx.symbol.squeeze(mx.sym.slice_axis(data=d_preds, axis=1, begin=i, end=i + 1), axis=1)  # shape, [N, num_parts, H, W]
                tmp_d_loss = mx.symbol.square(data=(tmp_d_pred - heatmaps))
                masks_expand = mx.symbol.expand_dims(masks, axis=1)
                tmp_d_loss = mx.symbol.broadcast_mul(lhs=tmp_d_loss, rhs=masks_expand)
                tmp_d_loss = mx.symbol.mean(data=tmp_d_loss, axis=(1, 2, 3))  # shape, [N]
                d_loss.append(tmp_d_loss)
            # stack all stage
            d_loss = mx.sym.stack(*d_loss, axis=1)  # shape, [N, num_stack]

            d_losses = mx.symbol.mean(data=d_loss, axis=0)  # shape, [num_stack]

            # d_losses = mx.symbol.Custom(op_type='monitor', input=d_losses, nickname='d_losses')

            '''
            a_loss = mx.symbol.Custom(preds=a_preds, keypoints=keypoints, op_type='a_loss',
                                      num_stack=num_stack, max_persons=max_persons, num_parts=num_parts)
            a_loss_inside = mx.symbol.squeeze(mx.sym.slice_axis(data=a_loss, axis=1, begin=0, end=1))
            a_loss_outside = mx.symbol.squeeze(mx.sym.slice_axis(data=a_loss, axis=1, begin=1, end=2))
            '''

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
            a_loss_inside_list = []
            a_loss_outside_list = []
            for i in range(num_stack):
                tmp_a_pred = mx.symbol.squeeze(mx.sym.slice_axis(data=a_preds, axis=1, begin=i, end=i + 1), axis=(1))

                # flatten
                tmp_a_pred = mx.sym.Reshape(data=tmp_a_pred, shape=(0, 0, -3), name='association_reshape{}'.format(i + 1))  # shape, [N, num_parts, feat_w * feat_h]
                person_mean_list = []

                # calc association loss inside one person
                tmp_a_loss_inside_list = []
                for j in range(max_persons):
                    # pick feat for person-j
                    tmp_person_feat = mx.sym.pick(tmp_a_pred, keypoints_idx_list[j])  # shape, [N, num_parts]
                    # calc person-j's mean feat
                    tmp_persin_feat_scale = mx.symbol.broadcast_div(keypoints_mask_list[j], keypoints_mask_list[j].sum(axis=1, keepdims=True) + (keypoints_mask_list[j].sum(axis=1, keepdims=True) == 0))  # shape, [N, num_parts]
                    tmp_person_feat_mean = mx.symbol.sum(tmp_person_feat * tmp_persin_feat_scale, axis=1)  # shape, [N]
                    # tmp_person_feat_mean = mx.symbol.Custom(op_type='monitor', input=tmp_person_feat_mean, nickname='tmp_person_feat_mean_stack{}_person{}'.format(i_j))

                    person_mean_list.append(tmp_person_feat_mean)
                    tmp_person_feat_mean_expand = mx.symbol.expand_dims(tmp_person_feat_mean, axis=1)  # shape, [N, 1]

                    # calc a_loss_inside
                    tmp_a_loss_inside = mx.symbol.square(data=mx.symbol.broadcast_sub(lhs=tmp_person_feat, rhs=tmp_person_feat_mean_expand))  # shape [N, num_parts]
                    tmp_a_loss_inside = mx.symbol.sum(tmp_a_loss_inside * tmp_persin_feat_scale, axis=1)  # shape, [N]
                    tmp_a_loss_inside_list.append(tmp_a_loss_inside)
                tmp_a_loss_inside = mx.sym.stack(*tmp_a_loss_inside_list, axis=1)  # shape, [N, max_persons]
                # tmp_a_loss_inside = mx.symbol.Custom(op_type='monitor', input=tmp_a_loss_inside, nickname='tmp_a_loss_inside_before_sum')
                tmp_a_loss_inside = mx.symbol.sum(tmp_a_loss_inside * tmp_a_loss_inside_scale, axis=1)  # shape, [N]
                # tmp_a_loss_inside = mx.symbol.Custom(op_type='monitor', input=tmp_a_loss_inside, nickname='tmp_a_loss_inside_after_sum')
                a_loss_inside_list.append(tmp_a_loss_inside)


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
                tmp_a_loss_outside_mean = mx.symbol.sum(tmp_a_loss_outside * tmp_a_loss_outside_scale, axis=1)  # shape, [N]
                a_loss_outside_list.append(tmp_a_loss_outside_mean)

            # stack all stage
            a_loss_inside = mx.sym.stack(*a_loss_inside_list, axis=1)  # shape, [N, num_stack]
            a_loss_outside = 0.5 * mx.sym.stack(*a_loss_outside_list, axis=1)  # shape, [N, num_stack]

            # a_loss_inside = mx.symbol.Custom(op_type='monitor', input=a_loss_inside, nickname='a_loss_inside')
            # a_loss_outside = mx.symbol.Custom(op_type='monitor', input=a_loss_outside, nickname='a_loss_outside')

            a_losses_inside = mx.symbol.mean(data=a_loss_inside, axis=0)  # shape, [num_stack]
            a_losses_outside = mx.symbol.mean(data=a_loss_outside, axis=0)  # shape, [num_stack]

            # d_losses = mx.symbol.Custom(op_type='monitor', input=d_losses, nickname='d_losses')

            # mask Loss
            d_loss = mx.sym.MakeLoss(name='detection_loss', data=mx.symbol.mean(data=d_losses), grad_scale=1.0)
            a_loss_inside = mx.sym.MakeLoss(name='association_loss_inside', data=mx.symbol.mean(data=a_losses_inside), grad_scale=0.001)
            a_loss_outside = mx.sym.MakeLoss(name='association_loss_outside', data=mx.symbol.mean(data=a_losses_outside), grad_scale=0.001)

            output_list = [d_loss, a_loss_inside, a_loss_outside]

            # get gpu metric
            if cfg.TRAIN.GPU_METRIC:
                for i in range(num_stack):
                    D_Loss = mx.sym.slice_axis(data=d_losses, axis=0, begin=i, end=i+1)
                    output_list.extend(get_detection_loss(D_Loss))
                    A_Loss_Inside = mx.sym.slice_axis(data=a_losses_inside, axis=0, begin=i, end=i+1)
                    output_list.extend(get_association_loss_inside(A_Loss_Inside))
                    A_Loss_Outside = mx.sym.slice_axis(data=a_losses_outside, axis=0, begin=i, end=i+1)
                    output_list.extend(get_association_loss_outside(A_Loss_Outside))
                output_list.extend(get_det_max(mx.symbol.squeeze(mx.sym.slice_axis(data=d_preds, axis=1, begin=3, end=4))))
                output_list.extend(get_tag_mean(mx.symbol.squeeze(mx.sym.slice_axis(data=a_preds, axis=1, begin=3, end=4))))
                output_list.extend(get_tag_var(mx.symbol.squeeze(mx.sym.slice_axis(data=a_preds, axis=1, begin=3, end=4))))
            else:
                raise ValueError('No CPU metric is supported now!')

            group = mx.sym.Group(output_list)
        else:
            group = mx.sym.Group([preds])

        self.sym = group
        return group

    def get_pred_names(self, is_train, gpu_metric=False):
        if is_train:
            pred_names = ['d_loss', 'a_loss_inside', 'a_loss_outside']
            if gpu_metric:
                pred_names.extend([
                    'D_Loss_0',
                    'A_Loss_Inside_0',
                    'A_Loss_Outside_0',
                    'D_Loss_1',
                    'A_Loss_Inside_1',
                    'A_Loss_Outside_1',
                    'D_Loss_2',
                    'A_Loss_Inside_2',
                    'A_Loss_Outside_2',
                    'D_Loss_3',
                    'A_Loss_Inside_3',
                    'A_Loss_Outside_3',
                    'Det_Max',
                    'Tag_Mean',
                    'Tag_Var'])
            return pred_names
        else:
            pred_names = ['d_loss', 'a_loss_inside', 'a_loss_outside']

    def get_label_names(self):
        return ['preds']

    def init_weight_pre(self, cfg, arg_params, aux_params):
        for ele in self.init_pre_list:
            if '_conv' in ele:
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
                # pytorch's kaiming_uniform_
                weight_shape = self.arg_shape_dict['{}_weight'.format(ele)]
                fan_in = float(weight_shape[1]) * weight_shape[2] * weight_shape[3]
                bound = np.sqrt(6 / ((1 + 5) * fan_in))
                arg_params['{}_weight'.format(ele)] = mx.random.uniform(-bound, bound, shape=weight_shape)
                arg_params['{}_bias'.format(ele)] = mx.random.uniform(-bound, bound, shape=self.arg_shape_dict['{}_bias'.format(ele)])

            elif '_relu' in ele:
                continue
            else:
                raise ValueError('Layer {} init not inplemented'.format(ele))

    def init_weight_hourglass(self, cfg, arg_params, aux_params):
        for ele in self.init_hourglass_list:
            if '_conv' in ele:
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
                # pytorch's kaiming_uniform_
                weight_shape = self.arg_shape_dict['{}_weight'.format(ele)]
                fan_in = float(weight_shape[1]) * weight_shape[2] * weight_shape[3]
                bound = np.sqrt(6 / ((1 + 5) * fan_in))
                arg_params['{}_weight'.format(ele)] = mx.random.uniform(-bound, bound, shape=weight_shape)
                arg_params['{}_bias'.format(ele)] = mx.random.uniform(-bound, bound, shape=self.arg_shape_dict['{}_bias'.format(ele)])

            elif '_relu' in ele:
                continue
            else:
                raise ValueError('Layer {} init not inplemented'.format(ele))

    def init_weight(self, cfg, arg_params, aux_params):
        self.init_weight_pre(cfg, arg_params, aux_params)
        self.init_weight_hourglass(cfg, arg_params, aux_params)
