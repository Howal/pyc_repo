import mxnet as mx
import numpy as np
from common.lib.utils.symbol import Symbol
from common.backbone import resnet_v1
from common.gpu_metric import *
from common.operator_py.monitor_op import *


class res101_fpn_pose(Symbol):
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


    def get_fpn_feature(self, c2, c3, c4, c5, feature_dim, use_dconv, dconv_lr_mult):
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
        if use_dconv:
            fpn_p6 = self.deformable_conv(c5, 'fpn_p6', feature_dim, stride=(2, 2), lr_mult=dconv_lr_mult)
            fpn_p4 = self.deformable_conv(fpn_p4_plus, 'fpn_p4', feature_dim, stride=(1, 1), lr_mult=dconv_lr_mult)
            fpn_p3 = self.deformable_conv(fpn_p3_plus, 'fpn_p3', feature_dim, stride=(1, 1), lr_mult=dconv_lr_mult)
            fpn_p2 = self.deformable_conv(fpn_p2_plus, 'fpn_p2', feature_dim, stride=(1, 1), lr_mult=dconv_lr_mult)
        else:
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

        # input init
        if is_train:
            data = mx.sym.Variable(name="data")  # img, [N, 3, H ,W]
            heatmaps = mx.sym.Variable(name="heatmaps")  # heatmaps of parts, [N, num_parts, H/4, W/4], REMARK 1/4 scale
            masks = mx.sym.Variable(name="masks")  # mask of crowds in coco, [N, H/4, W/4], REMARK 1/4 scale
            keypoint_visible = mx.sym.Variable(name='keypoint_visible')  # [N, max_persons, num_parts]
            keypoint_location = mx.sym.Variable(name='keypoint_location')  # [N, max_person, num_parts, 4]
            keypoint_location = mx.sym.transpose(keypoint_location, axes=(3, 0, 1, 2), name="keypoint_location_transpose")
        else:
            data = mx.sym.Variable(name="data")  # img, [N, 3, H ,W]

        assert cfg.network.use_dilation_on_c5 == False, "fpn should keep use_dilation_on_c5 to be False"
        _, c2, c3, c4, c5 = resnet_v1.get_resnet_backbone(data=data,
                                                          num_layers=cfg.network.num_layers,
                                                          use_dilation_on_c5=cfg.network.use_dilation_on_c5,
                                                          use_dconv=cfg.network.backbone_use_dconv,
                                                          dconv_lr_mult=cfg.network.backbone_dconv_lr_mult,
                                                          dconv_group=cfg.network.backbone_dconv_group,
                                                          dconv_start_channel=cfg.network.backbone_dconv_start_channel)

        fpn_p2, fpn_p3, fpn_p4, fpn_p5, fpn_p6 = self.get_fpn_feature(c2, c3, c4, c5,
                                                                      feature_dim=cfg.FPN.feature_dim,
                                                                      use_dconv=cfg.FPN.use_dconv,
                                                                      dconv_lr_mult=cfg.FPN.dconv_lr_mult)


        # fpn_p2 = mx.sym.Custom(op_type="monitor", data=fpn_p2, nickname="fpn_p2")
        det_pred = mx.sym.Convolution(data=fpn_p2, num_filter=num_parts, kernel=(1, 1), stride=(1, 1),
                                   no_bias=False, name='det_pred_1x1conv')  # shape, [N, num_parts, H, W]
        association_pred = mx.sym.Convolution(data=fpn_p2, num_filter=num_parts, kernel=(1, 1), stride=(1, 1),
                                   no_bias=False, name='association_pred_1x1conv')  # shape, [N, num_parts, H, W]

        # det_pred = mx.sym.Custom(op_type="monitor", data=det_pred, nickname="det_pred")
        # heatmaps = mx.sym.Custom(op_type="monitor", data=heatmaps, nickname="heatmaps")
        # calc_loss
        if is_train:
            # calc detection loss
            det_loss = mx.symbol.square(data=(det_pred - heatmaps))
            masks_4d = mx.symbol.expand_dims(masks, axis=1)
            det_loss = mx.symbol.broadcast_mul(det_loss, masks_4d).mean()
            # det_loss = mx.sym.Custom(op_type="monitor", data=det_loss, nickname="det_loss")

            # a_preds: [N, K, H, W]
            # a_pred:[N, K, H, W, 1]
            association_pred = association_pred.reshape(shape=(0, 0, 0, 0, 1))

            outside_loss, inside_loss = self.get_inside_outside_loss(feature=association_pred,
                                                                     keypoint_visible=keypoint_visible,
                                                                     keypoint_location=keypoint_location,
                                                                     batch_size=cfg.TRAIN.BATCH_IMAGES,
                                                                     num_keypoint_cls=num_parts,
                                                                     max_persons=max_persons,
                                                                     prefix="stack_1")
            # stack all stage
            inside_loss = mx.symbol.mean(data=inside_loss, axis=0)  # shape, [1]
            outside_loss = 0.5 * mx.symbol.mean(data=outside_loss, axis=0)  # shape, [1]

            # mask Loss
            det_loss = mx.sym.MakeLoss(name='detection_loss', data=det_loss, grad_scale=cfg.pose.det_loss_weight)
            inside_loss= mx.sym.MakeLoss(name='association_loss_inside', data=inside_loss, grad_scale=cfg.pose.inside_loss_weight)
            outside_loss = mx.sym.MakeLoss(name='association_loss_outside', data=outside_loss, grad_scale=cfg.pose.outside_loss_weight)

            output_list = [det_loss, inside_loss, outside_loss]

            # get gpu metric
            if cfg.TRAIN.GPU_METRIC:
                output_list.extend(get_detection_loss(det_loss))
                output_list.extend(get_association_loss_inside(inside_loss))
                output_list.extend(get_association_loss_outside(outside_loss))
                output_list.extend(get_det_max(det_pred))
                # output_list.extend(get_tag_mean(association_pred))
                # output_list.extend(get_tag_var(association_pred))
            else:
                raise ValueError('No CPU metric is supported now!')

            group = mx.sym.Group(output_list)
        else:
            group = mx.sym.Group([det_pred, association_pred])

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
                    'Det_Max'])
            return pred_names
        else:
            # pred_names = ['d_loss', 'a_loss_inside', 'a_loss_outside']
            raise NotImplementedError

    def get_label_names(self):
        return ['d_preds, a_preds']

    def init_pose_head(self, cfg, arg_params, aux_params):
        arg_params['det_pred_1x1conv_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['det_pred_1x1conv_weight'])
        arg_params['det_pred_1x1conv_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['det_pred_1x1conv_bias'])
        arg_params['association_pred_1x1conv_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['association_pred_1x1conv_weight'])
        arg_params['association_pred_1x1conv_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['association_pred_1x1conv_bias'])

    def init_weight_fpn(self, cfg, arg_params, aux_params):
        # arg_params['fpn_p6_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_p6_weight'])
        # arg_params['fpn_p6_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_p6_bias'])
        # arg_params['fpn_p5_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_p5_weight'])
        # arg_params['fpn_p5_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_p5_bias'])
        # arg_params['fpn_p4_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_p4_weight'])
        # arg_params['fpn_p4_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_p4_bias'])
        # arg_params['fpn_p3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_p3_weight'])
        # arg_params['fpn_p3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_p3_bias'])
        arg_params['fpn_p2_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_p2_weight'])
        arg_params['fpn_p2_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_p2_bias'])

        arg_params['fpn_p5_1x1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_p5_1x1_weight'])
        arg_params['fpn_p5_1x1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_p5_1x1_bias'])
        arg_params['fpn_p4_1x1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_p4_1x1_weight'])
        arg_params['fpn_p4_1x1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_p4_1x1_bias'])
        arg_params['fpn_p3_1x1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_p3_1x1_weight'])
        arg_params['fpn_p3_1x1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_p3_1x1_bias'])
        arg_params['fpn_p2_1x1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_p2_1x1_weight'])
        arg_params['fpn_p2_1x1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_p2_1x1_bias'])

    def init_weight_dcn_offset(self, cfg, arg_params, aux_params):
        for key in self.arg_shape_dict:
            if 'offset' in key and not key in arg_params:
                if 'reduce' in key and 'weight' in key:
                    arg_params[key] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict[key])
                else:
                    arg_params[key] = mx.nd.zeros(shape=self.arg_shape_dict[key])

    def init_weight(self, cfg, arg_params, aux_params):
        self.init_pose_head(cfg, arg_params, aux_params)
        self.init_weight_fpn(cfg, arg_params, aux_params)
        if cfg.network.backbone_use_dconv or cfg.FPN.use_dconv:
            self.init_weight_dcn_offset(cfg, arg_params, aux_params)