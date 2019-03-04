import mxnet as mx
import numpy as np
from common.lib.utils.symbol import Symbol
from common.backbone.hourglass_v1 import hourglass_v1, conv_sym_wrapper, CurrentBN
from common.gpu_metric import *
from common.operator_py.select_part import *
from common.operator_py.monitor_op import *

class posenet_v1_hourglass4_relation_v3(Symbol):
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


    # key_data: [N, Dim, num_part, K]
    # query_data: [N, Dim, H, W]
    def relation_module(self, key_data, query_data, affinity_dim, value_dim, output_dim, num_part, top_k, prefix=""):
        # query_embd: [N, Aff_Dim, H, W]
        query_embd = mx.sym.Convolution(query_data, kernel=(1,1), stride=(1,1), num_filter=affinity_dim, no_bias=True, name=prefix + "_query_embd")

        # key_embd: [N, Aff_Dim, num_part2, K]
        key_embd = mx.sym.Convolution(key_data, kernel=(1,1), stride=(1,1), num_filter=affinity_dim,
                                      no_bias=True, name=prefix + "_key_embd")

        # value_embd: [N, Val_Dim, num_part2, K]
        value_embd = mx.sym.Convolution(key_data, kernel=(1,1), stride=(1,1), num_filter=value_dim,
                                        no_bias=True, name=prefix + "_value_embd")

        # query_embd_reshape: [N, H*W, Aff_Dim]
        query_embd_reshape = query_embd.reshape(shape=(0, 0, -1)) #[N, Aff_dim, H*W]
        query_embd_reshape = mx.sym.transpose(query_embd_reshape , axes=(0, 2, 1)) #[N, H*W, Aff_dim]

        # key_embd_reshape: [N, num_part2 * K, Aff_Dim]
        key_embd_reshape = mx.sym.transpose(key_embd, axes=(0, 2, 3, 1)) #[N, num_part2, K, Aff_Dim]
        key_embd_reshape = key_embd_reshape.reshape(shape=(0, num_part*top_k, affinity_dim))

        # value_embd_reshape: [N * num_part2, K, Val_Dim]
        value_embd_reshape = mx.sym.transpose(value_embd, axes=(0, 2, 3, 1)) #[N, num_part2, K, Val_Dim]
        value_embd_reshape = value_embd_reshape.reshape(shape=(-1, top_k, value_dim), name=prefix +"_value_embd_reshape")

        # aff_mat = [N, H*W, num_part2*K]
        aff_mat = mx.sym.batch_dot(lhs=query_embd_reshape, rhs=key_embd_reshape, transpose_a=False, transpose_b=True, name=prefix + '_aff_mat_batch_dot')
        # aff_mat = [N, H*W, num_part2, K]
        aff_mat = aff_mat.reshape(shape=(0, 0, -4, num_part, top_k)) #
        # aff_mat_norm = [N, H*W, num_part2, K]
        aff_mat_norm = mx.sym.softmax(aff_mat, axis=3, name=prefix + "_aff_mat_softmax")
        # aff_mat_norm: [N, num_part2, H*W, K]
        aff_mat_norm = mx.sym.transpose(aff_mat_norm, axes=(0, 2, 1, 3))
        # aff_mat_norm: [N* num_part2, H*W, K]
        aff_mat_norm = mx.sym.reshape(aff_mat_norm, shape=(-3, 0, 0))

        # relation_feat: [N*num_part2, H*W, Val_Dim]
        relation_feat = mx.sym.batch_dot(lhs=aff_mat_norm, rhs=value_embd_reshape, transpose_a=False, transpose_b=False)
        # relation_feat: [N * num_part2, Val_dim, H*W]
        relation_feat = mx.sym.swapaxes(relation_feat, 1, 2)

        # relation_feat: [N, num_part2, Val_dim, H*W]
        relation_feat = mx.sym.reshape(relation_feat, shape=(-4, -1, num_part, value_dim, 0))
        # relation_feat: [N, num_part2 * Val_dim, H*W]
        relation_feat = mx.sym.reshape(relation_feat, shape=(0, -3, 0))
        # relation_feat: [N, output_dim, H*W]
        relation_feat = mx.sym.reshape(relation_feat, shape=(0, 0, 0, 1))
        relation_feat = mx.sym.Convolution(relation_feat, kernel=(1,1), stride=(1, 1), num_filter=output_dim, name=prefix + "_fusion")
        relation_feat = mx.sym.Activation(relation_feat, act_type='relu')
        # relation_feat: [N, output_dim, H * W]
        relation_feat = mx.sym.reshape_like(relation_feat, query_data)
        return relation_feat

    def get_stacked_hourglass(self, data, num_stack=4, in_dim=256, out_dim=68, increase_dim=128,
                              bn=CurrentBN(False, 0.9), record=[], num_parts=17, use_relation=[], cfg=None, is_train=False):

        preds = []
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
            preds.append(out)

            if use_relation[i]:
                top_k = cfg.pose.top_k
                prefix_name = "stack_{}".format(i)

                det_pred = mx.sym.slice_axis(data=out, axis=1, begin=0, end=num_parts)  # shape, [N, num_parts, H, W]
                pose_sensitive_feature = mx.sym.Convolution(data=feature, kernel=(1,1), stride=(1,1), num_filter=num_parts * cfg.pose.sensitive_dim, name=prefix_name + "_sensitve_conv")
                # select_part_indices = self.select_part(det_pred, feat_h, feat_w, 2, 50, prefix="head_{}".format(head_idx))
                select_part_indices = mx.sym.Custom(op_type="select_part", kernel=2, top_k=top_k, det_score=det_pred, name=prefix_name + "_select_part")
                # data_reshape: [N, num_parts, Dim, H, W]
                data_reshape = mx.sym.reshape(pose_sensitive_feature, shape=(0, -4, num_parts, -1, 0, 0))
                # data_reshape: [N, num_parts, H, W, Dim]
                data_reshape = mx.sym.transpose(data_reshape, axes=(0, 1, 3, 4, 2))
                # part_feat: [N, num_part, K, Dim]
                part_feat = mx.sym.gather_nd(data_reshape, indices=select_part_indices, name=prefix_name + "_gather_nd")
                # part_feat: [N, Dim, num_part, K]
                part_feat = mx.sym.transpose(part_feat, axes=(0, 3, 1, 2))
                # relation_feat: [N, num_parts*Dim, H, W]
                relation_feat = self.relation_module(key_data=part_feat, query_data=feature,
                                                     affinity_dim=cfg.pose.aff_dim, value_dim=cfg.pose.val_dim,
                                                     output_dim=in_dim,
                                                     num_part=num_parts, top_k=top_k, prefix=prefix_name)
                feature = feature + relation_feat

            if i != num_stack - 1:
                data_preds = conv_sym_wrapper(data=out, prefix="hg{}_merge_preds".format(i + 1),
                                              num_filter=in_dim, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                              bn=bn, relu=False, record=record)
                data_feats = conv_sym_wrapper(data=feature, prefix="hg{}_merge_feats".format(i + 1),
                                              num_filter=in_dim, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                              bn=bn, relu=False, record=record)
                data = data + data_preds + data_feats

        preds = mx.sym.stack(*preds, axis=1)
        return preds


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
        preds = self.get_stacked_hourglass(data=data, num_stack=num_stack, in_dim=in_dim, out_dim=out_dim,
                                           increase_dim=increase_dim, bn=bn, record=init_hourglass_list,
                                           num_parts=17, use_relation=cfg.pose.use_relation, cfg=cfg, is_train=is_train)
        self.init_hourglass_list = init_hourglass_list

        # preds = mx.sym.Custom(data=preds, op_type='monitor', nickname='preds')
        # calc_loss
        if is_train:
            # slice into two parts
            d_preds = mx.sym.slice_axis(data=preds, axis=2, begin=0, end=num_parts)  # shape, [N, num_stack, num_parts, H, W]
            a_preds = mx.sym.slice_axis(data=preds, axis=2, begin=num_parts, end=2*num_parts)  # shape, [N, num_stack, num_parts, H, W]

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

            # pick keypoint feats
            outside_loss_list = []
            inside_loss_list = []


            for i in range(num_stack):
                # a_pred: [N, K, H, W]
                a_pred = mx.symbol.squeeze(mx.sym.slice_axis(data=a_preds, axis=1, begin=i, end=i + 1), axis=(1))
                # a_pred = mx.sym.Custom(data=a_pred, op_type='monitor', nickname='stack_{}_a_pred'.format(i))
                # a_pred:[N, K, H, W, 1]
                a_pred = a_pred.reshape(shape=(0, 0, 0, 0, 1))
                # a_pred = mx.sym.Custom(data=a_pred, op_type='monitor', nickname='stack_{}_a_pred_reshape'.format(i))

                outside_loss, inside_loss = self.get_inside_outside_loss(feature=a_pred,
                                                                         keypoint_visible=keypoint_visible,
                                                                         keypoint_location=keypoint_location,
                                                                         batch_size=cfg.TRAIN.BATCH_IMAGES,
                                                                         num_keypoint_cls=17,
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
                for i in range(num_stack):
                    D_Loss = mx.sym.slice_axis(data=d_losses, axis=0, begin=i, end=i+1)
                    output_list.extend(get_detection_loss(D_Loss))
                    A_Loss_Inside = inside_loss_list[i]
                    output_list.extend(get_association_loss_inside(A_Loss_Inside))
                    A_Loss_Outside = outside_loss_list[i]
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
        else:
            pred_names = ['d_loss', 'a_loss_inside', 'a_loss_outside']

        return pred_names

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

    def init_weight_relation(self, cfg, arg_params, aux_params):
        weight_names = ['stack_{}_sensitve_conv', 'stack_{}_key_embd', 'stack_{}_value_embd', 'stack_{}_query_embd', 'stack_{}_fusion']
        bias_names = ['stack_{}_sensitve_conv', 'stack_{}_fusion']
        for i, flag in enumerate(cfg.pose.use_relation):
            if flag == False:
                continue
            for weight_name in weight_names:
                weight_shape = self.arg_shape_dict[(weight_name + '_weight').format(i)]
                fan_in = float(weight_shape[1]) * weight_shape[2] * weight_shape[3]
                bound = np.sqrt(6 / ((1 + 5) * fan_in))

                if cfg.pose.param_init == 'normal':
                    arg_params[(weight_name + '_weight').format(i)] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict[
                        (weight_name + '_weight').format(i)])
                else:
                    arg_params[(weight_name + '_weight').format(i)] = mx.random.uniform(-bound, bound, shape=weight_shape)
            for bias_name in bias_names:
                if cfg.pose.param_init == 'normal':
                    arg_params[(bias_name + '_bias').format(i)] = mx.nd.zeros(shape=self.arg_shape_dict[(bias_name + '_bias').format(i)])
                else:
                    arg_params[(bias_name + '_bias').format(i)] = mx.random.uniform(-bound, bound, self.arg_shape_dict[(bias_name + '_bias').format(i)])

    def init_weight(self, cfg, arg_params, aux_params):
        self.init_weight_pre(cfg, arg_params, aux_params)
        self.init_weight_hourglass(cfg, arg_params, aux_params)
        self.init_weight_relation(cfg, arg_params, aux_params)
