import mxnet as mx
import numpy as np
from common.lib.utils.symbol import Symbol
from common.backbone.hourglass_v1 import hourglass_v1, conv_sym_wrapper, CurrentBN
from common.gpu_metric import *
from common.operator_py.select_part import *
from common.operator_py.monitor_op import *

class posenet_v1_hourglass4_nonlocal_sqrtdim(Symbol):
    def __init__(self, FP16=False):
        """
        Use __init__ to define parameter network needs
        """
        # FP16 is not used for now
        self.FP16 = FP16
        self.init_pre_list = []
        self.init_hourglass_list = []
        self.cfg = None

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


    def get_non_local_feat(self, feature, aff_dim, val_dim, out_dim, prefix=""):
        # feature: [N, C, H, W]

        key_embd = mx.sym.Convolution(data=feature, kernel=(1,1), stride=(1,1), num_filter=aff_dim, no_bias=True, name=prefix + "_key_embed")
        query_embd = mx.sym.Convolution(data=feature, kernel=(1,1), stride=(1,1), num_filter=aff_dim, no_bias=True, name=prefix + "_query_embed")
        val_embd = mx.sym.Convolution(data=feature, kernel=(1,1), stride=(1,1), num_filter=val_dim, no_bias=True, name=prefix + "_val_embed")

        # key_embd_reshape: [N, H*W, aff_dim]
        key_embd_reshape = mx.sym.transpose(key_embd, axes=(0, 2, 3, 1))
        key_embd_reshape = mx.sym.reshape(key_embd_reshape, shape=(0, -1, aff_dim))

        # query_embd_reshape: [N, H*W, aff_dim]
        query_embd_reshape = mx.sym.transpose(query_embd, axes=(0, 2, 3, 1))
        query_embd_reshape = mx.sym.reshape(query_embd_reshape, shape=(0, -1, aff_dim))

        # val_embd_reshape: [N, H*W, val_dim]
        val_embd_reshape = mx.sym.transpose(val_embd, axes=(0, 2, 3, 1))
        val_embd_reshape = mx.sym.reshape(val_embd_reshape, shape=(0, -1, val_dim))

        # aff_mat: [N, H*W, H*W]
        aff_mat = mx.sym.batch_dot(key_embd_reshape, query_embd_reshape, transpose_a=False, transpose_b=True) / (aff_dim ** 0.5)
        aff_mat = mx.sym.softmax(aff_mat, axis=2)

        # relation_feat: [N, H*W, val_dim]
        relation_feat = mx.sym.batch_dot(aff_mat, val_embd_reshape)
        relation_feat = mx.sym.transpose(relation_feat, axes=(0,2,1))
        relation_feat = mx.sym.reshape(relation_feat, shape=(0,0,0,1))
        relation_feat = mx.sym.Convolution(relation_feat, kernel=(1,1), stride=(1,1), num_filter=out_dim, name=prefix + "_fusion")
        relation_feat = mx.sym.Activation(relation_feat, act_type='relu')
        relation_feat = mx.sym.reshape_like(relation_feat, feature)

        return relation_feat


    def get_stacked_hourglass(self, data, num_stack=4, in_dim=256, out_dim=68, increase_dim=128,
                              bn=CurrentBN(False, 0.9), record=[], num_parts=17, cfg=None, is_train=False):

        det_preds = []
        association_preds = []
        for i in range(num_stack):
            body = hourglass_v1(data=data,
                                num_stage=4,
                                in_dim=in_dim,
                                increase_dim=increase_dim,
                                bn=bn,
                                prefix="hg{}".format(i + 1),
                                record=record)
            body = conv_sym_wrapper(data=body, prefix="hg{}_out1".format(i + 1),
                                    num_filter=in_dim, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                    bn=bn, record=record)
            feature = conv_sym_wrapper(data=body, prefix="hg{}_out2".format(i + 1),
                                       num_filter=in_dim, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                       bn=bn, record=record)
            out = conv_sym_wrapper(data=feature, prefix="hg{}_out3".format(i + 1),
                                   num_filter=out_dim, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                   bn=bn, relu=False, record=record)
            #preds.append(out)
            d_pred = mx.sym.slice_axis(data=out, axis=1, begin=0, end=num_parts)  # shape, [N, num_stack, num_parts, H, W]
            a_pred = mx.sym.slice_axis(data=out, axis=1, begin=num_parts, end=2*num_parts)  # shape, [N, num_stack, num_parts, H, W]

            det_preds.append(d_pred)
            association_preds.append(a_pred)

            if i != num_stack - 1:
                data_preds = conv_sym_wrapper(data=out, prefix="hg{}_merge_preds".format(i + 1),
                                              num_filter=in_dim, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                              bn=bn, relu=False, record=record)
                data_feats = conv_sym_wrapper(data=feature, prefix="hg{}_merge_feats".format(i + 1),
                                              num_filter=in_dim, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                              bn=bn, relu=False, record=record)
                data = data + data_preds + data_feats


        for i in range(cfg.pose.head_num):
            prefix_name="head_{}".format(i)
            non_local_feat = self.get_non_local_feat(feature=feature, aff_dim=256, val_dim=256, out_dim=256, prefix=prefix_name)
            feature = mx.sym.concat(feature, non_local_feat, dim=1)
            feature = mx.sym.Convolution(data=feature, kernel=(1,1), stride=(1,1), num_filter=256, name=prefix_name+"_after_concat_1x1")

            d_pred =  mx.sym.Convolution(data=feature, kernel=(1,1), stride=(1,1), num_filter=num_parts, name=prefix_name+"_det")
            a_pred = mx.sym.Convolution(data=feature, kernel=(1, 1), stride=(1,1), num_filter=num_parts, name=prefix_name + "_association")

            det_preds.append(d_pred)
            association_preds.append(a_pred)

        return det_preds, association_preds

    def get_det_loss(self, det_pred, heatmaps, masks):
        det_loss = mx.symbol.square(data=(det_pred - heatmaps))
        masks_4d = mx.symbol.expand_dims(masks, axis=1)
        det_loss = mx.symbol.broadcast_mul(det_loss, masks_4d).mean()

        return det_loss


    def get_symbol(self, cfg, is_train=True):
        # config alias for convenient
        in_dim = 256
        increase_dim = 128
        out_dim = 68
        num_stack = 4
        self.cfg = cfg
        num_parts = cfg.dataset.NUM_PARTS
        max_persons = cfg.dataset.MAX_PERSONS

        # input init
        if is_train:
            data = mx.sym.Variable(name="data")  # img, [N, 3, H ,W]
            heatmaps = mx.sym.Variable(name="heatmaps")  # heatmaps of parts, [N, num_parts, H/4, W/4], REMARK 1/4 scale
            masks = mx.sym.Variable(name="masks")  # mask of crowds in coco, [N, H/4, W/4], REMARK 1/4 scale
            # keypoints = mx.sym.Variable(name='keypoints')  # coordinates of keypoints, [N, max_persons, num_parts, 2], REMARK 1/4 scale
            keypoint_visible = mx.sym.Variable(name='keypoint_visible') # [N, max_persons, num_parts]
            keypoint_location  = mx.sym.Variable(name='keypoint_location') # [N, max_person, num_parts, 4]
            keypoint_location = mx.sym.transpose(keypoint_location, axes=(3, 0, 1, 2), name="keypoint_location_transpose")
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
        det_preds, association_preds = self.get_stacked_hourglass(data=data, num_stack=num_stack, in_dim=in_dim, out_dim=out_dim,
                                           increase_dim=increase_dim, bn=bn, record=init_hourglass_list,
                                           num_parts=num_parts, cfg=cfg, is_train=is_train)
        self.init_hourglass_list = init_hourglass_list

        # preds = mx.sym.Custom(data=preds, op_type='monitor', nickname='preds')
        # calc_loss
        if is_train:
            # calc detection loss
            d_loss = []
            for i in range(len(det_preds)):
                d_loss.append(self.get_det_loss(det_preds[i], heatmaps, masks))


            # stack all stage
            d_loss = mx.sym.stack(*d_loss, axis=1)  # shape, [N, num_stack]

            d_losses = mx.symbol.mean(data=d_loss, axis=0)  # shape, [num_stack]

            # pick keypoint feats
            outside_loss_list = []
            inside_loss_list = []


            for i in range(len(association_preds)):
                # a_pred: [N, K, H, W]
                a_pred = association_preds[i]
                # a_pred = mx.sym.Custom(data=a_pred, op_type='monitor', nickname='stack_{}_a_pred'.format(i))
                # a_pred:[N, K, H, W, 1]
                a_pred = a_pred.reshape(shape=(0, 0, 0, 0, 1))
                # a_pred = mx.sym.Custom(data=a_pred, op_type='monitor', nickname='stack_{}_a_pred_reshape'.format(i))

                outside_loss, inside_loss = self.get_inside_outside_loss(feature=a_pred,
                                                                         keypoint_visible=keypoint_visible,
                                                                         keypoint_location=keypoint_location,
                                                                         batch_size=cfg.TRAIN.BATCH_IMAGES,
                                                                         num_keypoint_cls=num_parts,
                                                                         max_persons=max_persons,
                                                                         prefix="stack_{}".format(i))
                outside_loss_list.append(outside_loss * 0.5)
                inside_loss_list.append(inside_loss)

            outside_loss_all_stage = mx.sym.stack(*outside_loss_list)

            # stack all stage loss together
            outside_loss_all_stage = outside_loss_all_stage.mean()
            inside_loss_all_stage =  mx.sym.stack(*inside_loss_list).mean()


            outside_loss_all_stage = mx.sym.MakeLoss(outside_loss_all_stage, grad_scale=1e-3, name='outside_loss')
            inside_loss_all_stage = mx.sym.MakeLoss(inside_loss_all_stage, grad_scale=1e-3, name='inside_loss')
            det_loss_all_stage = mx.sym.MakeLoss(mx.symbol.mean(data=d_losses), grad_scale=1.0, name='det_loss')

            output_list = [det_loss_all_stage, inside_loss_all_stage, outside_loss_all_stage]

            # get gpu metric
            if cfg.TRAIN.GPU_METRIC:
                for i in range(num_stack + cfg.pose.head_num):
                    D_Loss = mx.sym.slice_axis(data=d_losses, axis=0, begin=i, end=i+1)
                    output_list.extend(get_detection_loss(D_Loss))
                    A_Loss_Inside = inside_loss_list[i]
                    output_list.extend(get_association_loss_inside(A_Loss_Inside))
                    A_Loss_Outside = outside_loss_list[i]
                    output_list.extend(get_association_loss_outside(A_Loss_Outside))
                output_list.extend(get_det_max(mx.symbol.squeeze(mx.sym.slice_axis(data=det_preds[-1], axis=1, begin=3, end=4))))
            else:
                raise ValueError('No CPU metric is supported now!')

            group = mx.sym.Group(output_list)
        else:
            det_final = mx.symbol.BlockGrad(det_preds[-1], name="det_final")
            association_final = mx.symbol.BlockGrad(association_preds[-1], name="association_final")

            group = mx.sym.Group([det_final, association_final])

        self.sym = group
        return group

    def get_pred_names(self, is_train, gpu_metric=False):
        if is_train:
            pred_names = ['d_loss', 'a_loss_inside', 'a_loss_outside']
            if gpu_metric:
                for i in range(4 + self.cfg.pose.head_num):
                    pred_names.append('D_Loss_{}'.format(i))
                    pred_names.append('A_Loss_Inside_{}'.format(i))
                    pred_names.append('A_Loss_Outside_{}'.format(i))

                pred_names.append('Det_Max')
            return pred_names
        else:
            pred_names = ['d_loss', 'a_loss_inside', 'a_loss_outside']

    def get_label_names(self):
        return ['preds']

    def init_weight_pre(self, cfg, arg_params, aux_params):
        for ele in self.init_pre_list:
            if '_conv' in ele:
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

    def init_weight_non_local(self, cfg, arg_params, aux_params):
        prefix_name = "head_{}"
        weight_names = ['key_embed', 'val_embed', 'query_embed', 'fusion', 'after_concat_1x1', 'det', 'association']
        bias_names = ['fusion', 'after_concat_1x1', 'det', 'association']
        for i in range(cfg.pose.head_num):
            for weight_name in weight_names:
                weight_name = prefix_name + "_" + weight_name
                weight_shape = self.arg_shape_dict[(weight_name + '_weight').format(i)]
                fan_in = float(weight_shape[1]) * weight_shape[2] * weight_shape[3]
                bound = np.sqrt(6 / ((1 + 5) * fan_in))

                if cfg.pose.param_init == 'normal':
                    arg_params[(weight_name + '_weight').format(i)] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict[
                        (weight_name + '_weight').format(i)])
                else:
                    arg_params[(weight_name + '_weight').format(i)] = mx.random.uniform(-bound, bound, shape=weight_shape)
            for bias_name in bias_names:
                bias_name = prefix_name + "_" + bias_name
                if cfg.pose.param_init == 'normal':
                    arg_params[(bias_name + '_bias').format(i)] = mx.nd.zeros(shape=self.arg_shape_dict[(bias_name + '_bias').format(i)])
                else:
                    arg_params[(bias_name + '_bias').format(i)] = mx.random.uniform(-bound, bound, self.arg_shape_dict[(bias_name + '_bias').format(i)])

    def init_weight(self, cfg, arg_params, aux_params):
        self.init_weight_pre(cfg, arg_params, aux_params)
        self.init_weight_hourglass(cfg, arg_params, aux_params)
        self.init_weight_non_local(cfg, arg_params, aux_params)
