import mxnet as mx
import numpy as np
from common.lib.utils.symbol import Symbol
from common.backbone import resnet_v1
from common.gpu_metric import *
from common.operator_py.a_loss import *
from common.operator_py.d_loss import *
from common.operator_py.monitor_op import *


class posenet_v1_resnet101_extra_4_conv(Symbol):
    def __init__(self, FP16=False):
        """
        Use __init__ to define parameter network needs
        """
        # FP16 is not used for now
        self.FP16 = FP16
        self.init_pre_list = []
        self.init_hourglass_list = []

    def get_inside_outside_loss(self, feature, keypoint_visible, keypoint_location, batch_size, max_persons, num_keypoint_cls, prefix):
        # visible_keypoint_num: [N, P, 1]
        visible_keypoint_num = keypoint_visible.sum(axis=2, keepdims=True)
        # visible_person: [N, P, 1]
        visible_person = visible_keypoint_num > 0
        # visible_person_t: [N, 1, P]
        visible_person_t = mx.sym.transpose(visible_person, axes=(0, 2, 1))
        # visible_person_pair: [N, P, P]
        eye_mat = mx.sym.eye(max_persons).reshape(shape=(1, max_persons, max_persons))
        visible_person_pair = mx.sym.broadcast_mul(mx.sym.broadcast_mul(visible_person, visible_person_t), 1-eye_mat)
        # visible_person_num: [N, 1, 1]
        visible_person_num = visible_person.sum(axis=1)

        # feature: [N, K, H, W, C]
        keypoint_feats = mx.sym.gather_nd(feature, keypoint_location)

        # keypoint_feat: [N, P, K, C]
        keypoint_feats = mx.sym.reshape(keypoint_feats, shape=(batch_size, -1, num_keypoint_cls, 0), name=prefix + "_keypoint_feats_reshape")

        keypoint_visible_4d = mx.sym.expand_dims(keypoint_visible, axis=3)

        # masked unvalid keypoint_feats
        keypoint_feats = mx.sym.broadcast_mul(keypoint_feats, keypoint_visible_4d, name='masked_keypoint_feats')

        # mean keypoint_feat: [N, P, C]
        mean_keypoint_feats = mx.sym.broadcast_div(mx.sym.sum(keypoint_feats, axis=2),
                                                   mx.sym.maximum(1, visible_keypoint_num))

        # mean keypoint_feat: [N, P, 1, C]
        mean_keypoint_feats = mx.sym.expand_dims(mean_keypoint_feats, axis=2)

        # calc outside loss
        # mean_keypoint_feats_t: [N, 1, P, C]
        mean_keypoint_feats_t = mx.sym.transpose(mean_keypoint_feats, axes=(0, 2, 1, 3))

        # mean_diff: [N, P, P, C]
        mean_sqr_diff = mx.sym.broadcast_sub(mean_keypoint_feats, mean_keypoint_feats_t, name=prefix + '_braodcast_sub_mean_sqr_diff')
        # mean_sqr_diff = mx.symbol.Custom(op_type='monitor', data=mean_sqr_diff, nickname='mean_sqr_diff')

        mean_sqr_diff = mx.sym.square(mean_sqr_diff).sum(axis=3)

        # visible_person_pair = mx.symbol.Custom(op_type='monitor', data=visible_person_pair, nickname='visible_person_pair')
        mean_sqr_diff = mean_sqr_diff * visible_person_pair
        # mean_sqr_diff = mx.symbol.Custom(op_type='monitor', data=mean_sqr_diff, nickname='mean_sqr_diff')

        # outside_loss: [N, P, P]
        outside_loss = mx.sym.exp(-mean_sqr_diff)
        # outside_loss = mx.symbol.Custom(op_type='monitor', data=outside_loss, nickname='outside_loss_before')
        outside_loss = outside_loss * visible_person_pair
        # outside_loss = mx.symbol.Custom(op_type='monitor', data=outside_loss, nickname='outside_loss')

        # outside_loss: [N, P*P]
        outside_loss = outside_loss.reshape(shape=(0, -1))
        # outside_loss: [N]
        norm_scale = mx.sym.maximum(1, mx.sym.square(visible_person_num) - visible_person_num).reshape(shape=(-1))
        outside_loss = outside_loss.sum(axis=1) / norm_scale
        # outside_loss = mx.symbol.Custom(op_type='monitor', data=outside_loss, nickname=prefix + '_outside_loss')

        # instance_diff_sqr: [N, P, K, 1]
        instance_sqr_diff = mx.sym.broadcast_sub(keypoint_feats, mean_keypoint_feats, name=prefix + '_broadcast_sub_instance_sqr_diff')
        # instance_sqr_diff = mx.symbol.Custom(op_type='monitor', data=instance_sqr_diff, nickname='instance_sqr_diff')
        instance_sqr_diff = mx.sym.square(instance_sqr_diff).sum(axis=3)

        instance_sqr_diff = instance_sqr_diff * keypoint_visible

        # inside loss
        inside_loss = instance_sqr_diff.sum(axis=2, keepdims=True) / mx.sym.maximum(1, visible_keypoint_num)
        inside_loss = inside_loss.sum(axis=1) / mx.sym.maximum(1, visible_person_num)

        outside_loss_mean = mx.sym.mean(outside_loss, name="outside_loss_mean")
        inside_loss_mean = mx.sym.mean(inside_loss, name="inside_loss_mean")

        return outside_loss_mean, inside_loss_mean

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
            data_b = mx.sym.Activation(data=data, act_type='relu', name=prefix + '_relu1')
            data_b = mx.sym.Convolution(data=data_b, num_filter=256, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                         no_bias=False, name=prefix + '_conv1')  # shape, [N, num_parts, H, W]
            data_b = mx.sym.Activation(data=data_b, act_type='relu', name=prefix + '_relu2')
            data_b = mx.sym.Convolution(data=data_b, num_filter=256, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                         no_bias=False, name=prefix + '_conv2')  # shape, [N, num_parts, H, W]
            data_b = mx.sym.Activation(data=data, act_type='relu', name=prefix + '_relu3')
            data_b = mx.sym.Convolution(data=data_b, num_filter=256, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                         no_bias=False, name=prefix + '_conv3')  # shape, [N, num_parts, H, W]
            data_b = mx.sym.Activation(data=data_b, act_type='relu', name=prefix + '_relu4')
            data_b = mx.sym.Convolution(data=data_b, num_filter=256, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                         no_bias=False, name=prefix + '_conv4')  # shape, [N, num_parts, H, W]
            data = data + data_b
            data = mx.sym.Activation(data=data, act_type='relu', name=prefix + '_relu5')


        d_preds = mx.sym.Convolution(data=data, num_filter=num_parts, kernel=(1, 1), stride=(1, 1),
                                   no_bias=False, name='simple_baseline_d_preds')  # shape, [N, num_parts, H, W]
        a_preds = mx.sym.Convolution(data=data, num_filter=num_parts, kernel=(1, 1), stride=(1, 1),
                                   no_bias=False, name='simple_baseline_a_preds')  # shape, [N, num_parts, H, W]

        # calc_loss
        if is_train:
            # calc detection loss
            tmp_d_loss = mx.symbol.square(data=(d_preds - heatmaps))
            masks_expand = mx.symbol.expand_dims(masks, axis=1)
            tmp_d_loss = mx.symbol.broadcast_mul(lhs=tmp_d_loss, rhs=masks_expand)
            tmp_d_loss = mx.symbol.mean(data=tmp_d_loss, axis=(1, 2, 3))  # shape, [N]

            d_loss = mx.symbol.mean(data=tmp_d_loss, axis=0)  # shape, [1]


            # a_preds: [N, K, H, W]
            # a_pred:[N, K, H, W, 1]
            a_pred = a_preds.reshape(shape=(0, 0, 0, 0, 1))

            tmp_a_loss_outside, tmp_a_loss_inside = self.get_inside_outside_loss(feature=a_pred,
                                                                                 keypoint_visible=keypoint_visible,
                                                                                 keypoint_location=keypoint_location,
                                                                                 batch_size=cfg.TRAIN.BATCH_IMAGES,
                                                                                 num_keypoint_cls=17,
                                                                                 max_persons=max_persons,
                                                                                 prefix="stack_1")
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

            weight_shape = self.arg_shape_dict[prefix + '_conv1_weight']
            fan_in = float(weight_shape[1]) * weight_shape[2] * weight_shape[3]
            bound = np.sqrt(6 / ((1 + 5) * fan_in))
            arg_params[prefix + '_conv1_weight'] = mx.random.uniform(-bound, bound, shape=weight_shape)
            arg_params[prefix + '_conv1_bias'] = mx.random.uniform(-bound, bound, shape=self.arg_shape_dict[prefix + '_conv1_bias'])

            weight_shape = self.arg_shape_dict[prefix + '_conv2_weight']
            fan_in = float(weight_shape[1]) * weight_shape[2] * weight_shape[3]
            bound = np.sqrt(6 / ((1 + 5) * fan_in))
            arg_params[prefix + '_conv2_weight'] = mx.random.uniform(-bound, bound, shape=weight_shape)
            arg_params[prefix + '_conv2_bias'] = mx.random.uniform(-bound, bound, shape=self.arg_shape_dict[prefix + '_conv2_bias'])

            weight_shape = self.arg_shape_dict[prefix + '_conv3_weight']
            fan_in = float(weight_shape[1]) * weight_shape[2] * weight_shape[3]
            bound = np.sqrt(6 / ((1 + 5) * fan_in))
            arg_params[prefix + '_conv3_weight'] = mx.random.uniform(-bound, bound, shape=weight_shape)
            arg_params[prefix + '_conv3_bias'] = mx.random.uniform(-bound, bound, shape=self.arg_shape_dict[prefix + '_conv3_bias'])

            weight_shape = self.arg_shape_dict[prefix + '_conv4_weight']
            fan_in = float(weight_shape[1]) * weight_shape[2] * weight_shape[3]
            bound = np.sqrt(6 / ((1 + 5) * fan_in))
            arg_params[prefix + '_conv4_weight'] = mx.random.uniform(-bound, bound, shape=weight_shape)
            arg_params[prefix + '_conv4_bias'] = mx.random.uniform(-bound, bound, shape=self.arg_shape_dict[prefix + '_conv4_bias'])

            # arg_params[prefix + '_bn_gamma'] = mx.random.uniform(0, 1, shape=self.arg_shape_dict[prefix + '_bn_gamma'])
            # arg_params[prefix + '_bn_beta'] = mx.nd.zeros(shape=self.arg_shape_dict[prefix + '_bn_beta'])
            # aux_params[prefix + '_bn_moving_mean'] = mx.nd.zeros(shape=self.aux_shape_dict[prefix + '_bn_moving_mean'])
            # aux_params[prefix + '_bn_moving_var'] = mx.nd.ones(shape=self.aux_shape_dict[prefix + '_bn_moving_var'])

        # pytorch's kaiming_uniform_
        weight_shape = self.arg_shape_dict['simple_baseline_d_preds_weight']
        fan_in = float(weight_shape[1]) * weight_shape[2] * weight_shape[3]
        bound = np.sqrt(6 / ((1 + 5) * fan_in))
        arg_params['simple_baseline_d_preds_weight'] = mx.random.uniform(-bound, bound, shape=weight_shape)
        arg_params['simple_baseline_d_preds_bias'] = mx.random.uniform(-bound, bound, shape=self.arg_shape_dict['simple_baseline_d_preds_bias'])
        arg_params['simple_baseline_a_preds_weight'] = mx.random.uniform(-bound, bound, shape=self.arg_shape_dict['simple_baseline_a_preds_weight'])
        arg_params['simple_baseline_a_preds_bias'] = mx.random.uniform(-bound, bound, shape=self.arg_shape_dict['simple_baseline_a_preds_bias'])

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
