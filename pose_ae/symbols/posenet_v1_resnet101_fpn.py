import mxnet as mx
import numpy as np
from common.lib.utils.symbol import Symbol
from common.backbone import resnet_v1
from common.gpu_metric import *
from common.operator_py.a_loss import *
from common.operator_py.d_loss import *
from common.operator_py.monitor_op import *


class posenet_v1_resnet101_fpn(Symbol):
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
        mean_sqr_diff = mx.sym.square(mean_sqr_diff).sum(axis=3)

        # outside_loss: [N, P, P]
        outside_loss = mx.sym.exp(-mean_sqr_diff)
        outside_loss = outside_loss * visible_person_pair

        # outside_loss: [N, P*P]
        outside_loss = outside_loss.reshape(shape=(0, -1))
        # outside_loss: [N]
        norm_scale = mx.sym.maximum(1, mx.sym.square(visible_person_num) - visible_person_num).reshape(shape=(-1))
        outside_loss = outside_loss.sum(axis=1) / norm_scale
        # outside_loss = mx.symbol.Custom(op_type='monitor', data=outside_loss, nickname=prefix + '_outside_loss')

        # instance_diff_sqr: [N, P, K, 1]
        instance_sqr_diff = mx.sym.broadcast_sub(keypoint_feats, mean_keypoint_feats, name=prefix + '_broadcast_sub_instance_sqr_diff')
        instance_sqr_diff = mx.sym.square(instance_sqr_diff).sum(axis=3)

        instance_sqr_diff = instance_sqr_diff * keypoint_visible

        # inside loss
        inside_loss = instance_sqr_diff.sum(axis=2, keepdims=True) / mx.sym.maximum(1, visible_keypoint_num)
        inside_loss = inside_loss.sum(axis=1) / mx.sym.maximum(1, visible_person_num)

        outside_loss_mean = mx.sym.mean(outside_loss, name="outside_loss_mean")
        inside_loss_mean = mx.sym.mean(inside_loss, name="inside_loss_mean")

        return outside_loss_mean, inside_loss_mean

    def get_fpn_feature(self, c2, c3, c4, c5, feature_dim):
        # lateral connection
        fpn_p5_1x1 = mx.symbol.Convolution(data=c5, kernel=(1, 1), pad=(0, 0), stride=(1, 1), num_filter=feature_dim, name='fpn_p5_1x1')
        fpn_p4_1x1 = mx.symbol.Convolution(data=c4, kernel=(1, 1), pad=(0, 0), stride=(1, 1), num_filter=feature_dim, name='fpn_p4_1x1')
        fpn_p3_1x1 = mx.symbol.Convolution(data=c3, kernel=(1, 1), pad=(0, 0), stride=(1, 1), num_filter=feature_dim, name='fpn_p3_1x1')
        fpn_p2_1x1 = mx.symbol.Convolution(data=c2, kernel=(1, 1), pad=(0, 0), stride=(1, 1), num_filter=feature_dim, name='fpn_p2_1x1')
        # top-down connection
        fpn_p5_upsample = mx.symbol.UpSampling(fpn_p5_1x1, scale=2, sample_type='nearest', name='fpn_p5_upsample')
        fpn_p4_plus = mx.sym.ElementWiseSum(*[fpn_p5_upsample, fpn_p4_1x1], name='fpn_p4_sum')
        fpn_p4_upsample = mx.symbol.UpSampling(fpn_p4_plus, scale=2, sample_type='nearest', name='fpn_p4_upsample')
        fpn_p3_plus = mx.sym.ElementWiseSum(*[fpn_p4_upsample, fpn_p3_1x1], name='fpn_p3_sum')
        fpn_p3_upsample = mx.symbol.UpSampling(fpn_p3_plus, scale=2, sample_type='nearest', name='fpn_p3_upsample')
        fpn_p2_plus = mx.sym.ElementWiseSum(*[fpn_p3_upsample, fpn_p2_1x1], name='fpn_p2_sum')
        # FPN feature
        fpn_p6 = mx.symbol.Convolution(data=c5, kernel=(3, 3), pad=(1, 1), stride=(2, 2), num_filter=feature_dim, name='fpn_p6')
        fpn_p5 = mx.symbol.Convolution(data=fpn_p5_1x1, kernel=(3, 3), pad=(1, 1), stride=(1, 1), num_filter=feature_dim, name='fpn_p5')
        fpn_p4 = mx.symbol.Convolution(data=fpn_p4_plus, kernel=(3, 3), pad=(1, 1), stride=(1, 1), num_filter=feature_dim, name='fpn_p4')
        fpn_p3 = mx.symbol.Convolution(data=fpn_p3_plus, kernel=(3, 3), pad=(1, 1), stride=(1, 1), num_filter=feature_dim, name='fpn_p3')
        fpn_p2 = mx.symbol.Convolution(data=fpn_p2_plus, kernel=(3, 3), pad=(1, 1), stride=(1, 1), num_filter=feature_dim, name='fpn_p2')

        return fpn_p2, fpn_p3, fpn_p4, fpn_p5, fpn_p6

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
            keypoint_visible = mx.sym.Variable(name='keypoint_visible')  # [N, max_persons, num_parts]
            keypoint_location = mx.sym.Variable(name='keypoint_location')  # [N, max_person, num_parts, 4]
            keypoint_location = mx.sym.transpose(keypoint_location, axes=(3, 0, 1, 2), name="keypoint_location_transpose")
            bn_use_global_stats = False
        else:
            data = mx.sym.Variable(name="data")  # img, [N, 3, H ,W]
            bn_use_global_stats = True

        _, c2, c3, c4, c5 = resnet_v1.get_resnet_backbone(data=data, num_layers=101,
                                                       use_dilation_on_c5=False,
                                                       use_dconv=False, dconv_lr_mult=0.001, dconv_group=1, dconv_start_channel=512,
                                                       bn_mom=0.9)
        # FPN p2 shape W,H = 128, 128
        fpn_p2, _, _, _, _ = self.get_fpn_feature(c2, c3, c4, c5, feature_dim=256)
        data = fpn_p2

        # data = mx.sym.Convolution(data=data, num_filter=256, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
        #                           no_bias=False, name='preds_conv0')
        # data = mx.sym.Activation(data=data, act_type='relu', name='preds_relu0')

        d_preds = mx.sym.Convolution(data=data, num_filter=num_parts, kernel=(1, 1), stride=(1, 1),
                                   no_bias=False, name='d_preds_conv0')  # shape, [N, num_parts, H, W]
        a_preds = mx.sym.Convolution(data=data, num_filter=num_parts, kernel=(1, 1), stride=(1, 1),
                                   no_bias=False, name='a_preds_conv0')  # shape, [N, num_parts, H, W]

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

    def init_weight_fpn(self, cfg, arg_params, aux_params):
        for _stage in range(2, 6):
            prefix = 'fpn_p{}_1x1'.format(_stage)
            arg_params[prefix + '_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict[prefix + '_weight'])
            arg_params[prefix + '_bias'] = mx.nd.zeros(shape=self.arg_shape_dict[prefix + '_bias'])
        for _stage in range(2, 7):
            prefix = 'fpn_p{}'.format(_stage)
            if prefix+'_weight' in self.arg_shape_dict.keys():
                arg_params[prefix + '_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict[prefix + '_weight'])
                arg_params[prefix + '_bias'] = mx.nd.zeros(shape=self.arg_shape_dict[prefix + '_bias'])

        # # pytorch's kaiming_uniform_
        # weight_shape = self.arg_shape_dict['preds_conv0_weight']
        # fan_in = float(weight_shape[1]) * weight_shape[2] * weight_shape[3]
        # bound = np.sqrt(6 / ((1 + 5) * fan_in))
        # arg_params['preds_conv0_weight'] = mx.random.uniform(-bound, bound, shape=weight_shape)
        # arg_params['preds_conv0_bias'] = mx.random.uniform(-bound, bound, shape=self.arg_shape_dict['preds_conv0_bias'])

        # pytorch's kaiming_uniform_
        weight_shape = self.arg_shape_dict['d_preds_conv0_weight']
        fan_in = float(weight_shape[1]) * weight_shape[2] * weight_shape[3]
        bound = np.sqrt(6 / ((1 + 5) * fan_in))
        arg_params['d_preds_conv0_weight'] = mx.random.uniform(-bound, bound, shape=weight_shape)
        arg_params['d_preds_conv0_bias'] = mx.random.uniform(-bound, bound, shape=self.arg_shape_dict['d_preds_conv0_bias'])

        # a_preds branch's init
        arg_params['a_preds_conv0_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['a_preds_conv0_weight'])
        arg_params['a_preds_conv0_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['a_preds_conv0_bias'])

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
        self.init_weight_fpn(cfg, arg_params, aux_params)
