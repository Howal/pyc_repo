import mxnet as mx
import numpy as np
from common.lib.utils.symbol import Symbol
from common.backbone.hourglass_v1 import hourglass_v1, conv_sym_wrapper, CurrentBN
from common.gpu_metric import *
from common.operator_py.select_part import *
from common.operator_py.monitor_op import *
from common.operator_py.monitor_op_multivar import *
from common.operator_py.draw_attention_map import *
from common.operator_py.aff_prob_gt import *
from common.operator_py.neg_keypoint_sampler import *

from relation_helper_sym import *

class posenet_v1_hourglass4_relation_cat_softmax(Symbol):
    def __init__(self, FP16=False):
        """
        Use __init__ to define parameter network needs
        """
        # FP16 is not used for now
        self.FP16 = FP16
        self.init_pre_list = []
        self.init_hourglass_list = []
        self.cfg = None

    # key_data: [N, num_parts, Dim, H, W]
    # query_data: [N, Dim, H, W]
    def relation_module(self, key_data, query_data, select_part_indices, affinity_dim, value_dim, output_dim, num_part, top_k,
                        prefix=""):

        # query_embd: [N, Aff_Dim, H, W]
        query_embd = mx.sym.Convolution(query_data, kernel=(1, 1), stride=(1, 1), num_filter=affinity_dim,
                                        no_bias=True, name=prefix + "_query_embed")
        # query_embd_reshape: [N, H*W, Aff_Dim]
        query_embd_reshape = mx.sym.transpose(query_embd, axes=(0, 2, 3, 1)) #query_embd_reshape: [N, H, W, Aff_Dim]
        query_embd_reshape = query_embd_reshape.reshape(shape=(0, -3, 0)) # query_embd_reshape: [N, H*W, Aff_Dim]

        # key_data_reshape: [N * num_parts, Dim, H, W]
        key_data_reshape = key_data.reshape(shape=(-3, 0, 0, 0), name=prefix + "key_data_reshape")

        # key_embd: [N * num_parts, Aff_Dim, H, W]
        key_embd = mx.sym.Convolution(key_data_reshape, kernel=(1, 1), stride=(1, 1), num_filter=affinity_dim,
                                      no_bias=True, name=prefix + "_key_embed")

        # value_embd: [N * num_parts, Val_Dim, H, W]
        value_embd = mx.sym.Convolution(key_data_reshape, kernel=(1, 1), stride=(1, 1), num_filter=value_dim,
                                        no_bias=True, name=prefix + "_val_embed")


        # key_embd_reshape: [N, num_parts, H, W, Aff_Dim]
        key_embd_reshape = key_embd.reshape(shape=(-4, -1, num_part, 0, 0, 0)) # key_embd_reshape: [N, num_parts, Aff_Dim, H, W]
        key_embd_reshape = mx.sym.transpose(key_embd_reshape, axes=(0, 1, 3, 4, 2))

        # part_key_embd: [N, num_parts * K, Aff_Dim]
        part_key_embd =  mx.sym.gather_nd(key_embd_reshape, indices=mx.sym.BlockGrad(select_part_indices), name=prefix + "_gather_part_feat") # part_key_embd:  [N, num_part, K, Aff_Dim]
        part_key_embd = part_key_embd.reshape(shape=(0, -3, 0))


        # val_embd_reshape: [N, num_parts, H, W, Val_Dim]
        val_embd_reshape = value_embd.reshape(shape=(-4, -1, num_part, 0, 0, 0)) # key_embd_reshape: [N, num_parts, Aff_Dim, H, W]
        val_embd_reshape = mx.sym.transpose(val_embd_reshape, axes=(0, 1, 3, 4, 2))

        # part_val_embd: [N * num_parts, K, Val_dim]
        part_val_embd =  mx.sym.gather_nd(val_embd_reshape, indices=mx.sym.BlockGrad(select_part_indices), name=prefix + "_gather_part_feat") # part_key_embd:  [N, num_part, K, Val_dim]
        part_val_embd = part_val_embd.reshape(shape=(-3, 0, 0))


        # query_part_aff: [N, H*W, num_parts, K]
        query_part_aff = mx.sym.batch_dot(query_embd_reshape, part_key_embd, transpose_a=False, transpose_b=True) #query_part_aff: [N, H*W, num_parts*K]
        query_part_aff = query_part_aff.reshape(shape=(0, 0, num_part, top_k))

        # query_self_aff: [N, H*W, num_parts, 1]
        key_embd_reshape_2 = key_embd_reshape.reshape(shape=(0, 0, -3, 0)) #key_embd_reshape: [N, num_parts, H*W, Aff_Dim]
        query_embd_reshape_2 = query_embd_reshape.expand_dims(axis=1) #query_embd_reshape: [N, 1, H*W, Aff_Dim]
        query_self_aff = mx.sym.broadcast_mul(key_embd_reshape_2, query_embd_reshape_2) #query_self_aff: [N, num_parts, H*W, Aff_Dim]
        query_self_aff = query_self_aff.sum(axis=-1, keepdims=True) # query_self_aff: [N, num_parts, H*W, 1]
        query_self_aff = query_self_aff.swapaxes(1, 2) #query_self_aff: [N, H*W, num_parts, 1]

        # aff_mat_softmax: [N * num_parts, H*W, K+1]
        aff_mat = mx.sym.concat(query_part_aff, query_self_aff, dim=3) # aff_mat: [N, H*W, num_parts, K+1]
        aff_mat_transpose= mx.sym.swapaxes(aff_mat, 1, 3) # aff_mat_transpose: [N, K+1, num_parts, H*W]
        aff_mat_softmax = mx.sym.SoftmaxActivation(aff_mat_transpose, mode='channel') #aff_mat_softmax: [N, K+1,num_pafts, H*W]
        aff_mat_softmax = mx.sym.transpose(aff_mat_softmax, axes=(0,2, 3, 1)) #aff_mat_softmax: [N, num_parts, H*W, K+1]
        aff_mat_softmax = aff_mat_softmax.reshape(shape=(-3, 0, 0))

        # aff_mat_softmax_ret: [N, H, W, num_parts, K+1]
        aff_mat_softmax_ret = aff_mat_softmax.reshape(shape=(-4, -1, num_part, -4, 128, 128, 0)) #aff_mat_softmax_ret: [N, num_parts, H, W, K+1]
        aff_mat_softmax_ret = mx.sym.transpose(aff_mat_softmax_ret, axes=(0, 2, 3, 1, 4)) #aff_mat_softmax_ret: [N, H, W, num_parts, k+1]


        # part_aff_mat_softmax: [N * num_parts, H*W, K]
        part_aff_mat_softmax = mx.sym.slice_axis(aff_mat_softmax, axis=2, begin=0, end=top_k)

        # value_from_part: [N * num_parts, H*W, val_dim]
        value_from_part = mx.sym.batch_dot(part_aff_mat_softmax, part_val_embd, transpose_a=False, transpose_b=False)

        # self_aff_mat_softmax: [N * num_parts, H*W, 1]
        self_aff_mat_softmax = mx.sym.slice_axis(aff_mat_softmax, axis=2, begin=top_k, end=top_k+1)

        # value_from_self: [N * num_parts, H*W, val_dim]
        val_embd_reshape_2 = val_embd_reshape.reshape(shape=(-3, -3, 0)) # val_embd_reshape: [N*num_parts, H*W, val_dim]
        value_from_self = mx.sym.broadcast_mul(val_embd_reshape_2, self_aff_mat_softmax)

        # relation_feat: [N, num_parts * val_dim, H * W, 1]
        relation_feat = value_from_part + value_from_self # relation_feat: [N * num_port, H*W, val_dim]
        relation_feat = relation_feat.swapaxes(1, 2) # swapaxes: [N * num_parts, val_dim, H*W]
        relation_feat = relation_feat.reshape(shape=(-4, -1, num_part, 0, 0)) # relation_feat: [N, num_parts, val_dim, H*W]
        relation_feat = relation_feat.reshape(shape=(0, -3, 0, 1)) # relation_feat: [N, num_parts*val_dim, H*W, 1]

        # relation_feat: [N, output_dim, H*W]
        relation_feat = mx.sym.Convolution(relation_feat, kernel=(1, 1), stride=(1, 1), num_filter=output_dim,
                                           name=prefix + "_fusion")
        relation_feat = mx.sym.Activation(relation_feat, act_type='relu')
        # relation_feat: [N, output_dim, H * W]
        relation_feat = mx.sym.reshape_like(relation_feat, query_data)

        return relation_feat, aff_mat_softmax_ret

    def get_stacked_hourglass(self, data, keypoint_location, keypoint_visible, num_stack=4, in_dim=256, out_dim=68, increase_dim=128,
                              bn=CurrentBN(False, 0.9), record=[], num_parts=17, cfg=None, is_train=False, im=None):

        det_preds = []
        association_preds = []
        aff_losses = []
        aff_labels = []
        for i in range(num_stack):
            body = hourglass_v1(data=data, num_stage=4, in_dim=in_dim, increase_dim=increase_dim, bn=bn, prefix="hg{}".format(i + 1), record=record)
            body = conv_sym_wrapper(data=body, prefix="hg{}_out1".format(i + 1), num_filter=in_dim, kernel=(3, 3), stride=(1, 1), pad=(1, 1), bn=bn, record=record)
            feature = conv_sym_wrapper(data=body, prefix="hg{}_out2".format(i + 1), num_filter=in_dim, kernel=(3, 3), stride=(1, 1), pad=(1, 1), bn=bn, record=record)
            out = conv_sym_wrapper(data=feature, prefix="hg{}_out3".format(i + 1), num_filter=out_dim, kernel=(1, 1), stride=(1, 1), pad=(0, 0), bn=bn, relu=False, record=record)
            d_pred = mx.sym.slice_axis(data=out, axis=1, begin=0, end=num_parts)  # shape, [N, num_parts, H, W]
            a_pred = mx.sym.slice_axis(data=out, axis=1, begin=num_parts, end=2*num_parts)  # shape, [N, num_parts, H, W]

            det_preds.append(d_pred)
            association_preds.append(a_pred)

            if i != num_stack - 1:
                data_preds = conv_sym_wrapper(data=out, prefix="hg{}_merge_preds".format(i + 1), num_filter=in_dim, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                              bn=bn, relu=False, record=record)
                data_feats = conv_sym_wrapper(data=feature, prefix="hg{}_merge_feats".format(i + 1), num_filter=in_dim, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                              bn=bn, relu=False, record=record)
                data = data + data_preds + data_feats


        for i in range(cfg.pose.head_num):
            prefix_name="head_{}".format(i)

            top_k = cfg.pose.top_k

            # [N, num_parts * Dim, H, W]
            pose_sensitive_feature = mx.sym.Convolution(data=feature, kernel=(1, 1), stride=(1, 1),
                                                        num_filter=num_parts * cfg.pose.sensitive_dim,
                                                        name=prefix_name + "_sensitve_conv")
            select_part_indices = mx.sym.Custom(op_type="select_part", kernel=cfg.pose.nms, top_k=top_k,
                                                det_score=mx.sym.BlockGrad(d_pred),
                                                name=prefix_name + "_select_part")
            # data_reshape: [N, num_parts, Dim, H, W]
            data_reshape = mx.sym.reshape(pose_sensitive_feature, shape=(0, num_parts, cfg.pose.sensitive_dim, 128, 128))

            # relation_feat: [N, num_parts*Dim, H, W]
            # aff_mat_norm = [N, H, W, num_parts, K+1]
            relation_feat, aff_mat_norm = self.relation_module(key_data=data_reshape, query_data=feature,
                                                             select_part_indices = select_part_indices,
                                                             affinity_dim=cfg.pose.aff_dim, value_dim=cfg.pose.val_dim,
                                                             output_dim=in_dim,
                                                             num_part=num_parts, top_k=top_k, prefix=prefix_name)

            # aff_mat_norm_reshape: [N, H, W, num_parts * (K+1)]
            aff_mat_norm_reshape = aff_mat_norm.reshape(shape=(0, 0, 0, -1))

            # aff_gt: [N, gt_person * num_part, num_part, k+1]
            aff_prob_gt = mx.sym.Custom(op_type="aff_prob_gt", keypoint_location=mx.sym.BlockGrad(keypoint_location),
                                   keypoint_visible=mx.sym.BlockGrad(keypoint_visible), det_score=mx.sym.BlockGrad(d_pred),
                                   select_part_indices=mx.sym.BlockGrad(select_part_indices), radius=cfg.pose.fg_radius,
                                   contain_self = True, debug=True)

            valid_aff_gt = aff_prob_gt >= 0
            aff_prob_gt = aff_prob_gt * valid_aff_gt

            # agg_keypoint_location: [3, N, gt_person * num_part]
            agg_keypoint_location = get_aggregate_keypoint_loc(keypoint_location)
            # fg_aff: [N, gt_person*num_part, num_part*K+1]
            fg_aff = mx.sym.gather_nd(aff_mat_norm_reshape, indices=agg_keypoint_location, name=prefix_name + "_gather_fg_aff")
            # fg_aff_reshape: [N, gt_person*num_part, num_part, K+1]
            fg_aff_reshape = mx.sym.reshape(fg_aff, shape=(0,0, num_parts, top_k+1))

            # masked_fg_aff: [N, gt_person*num_part, num_part, K+1]
            masked_fg_aff = mx.sym.broadcast_mul(fg_aff_reshape, valid_aff_gt)

            # aff_mat_reshape = aff_mat_norm_reshape
            # im, d_pred, aff_mat_reshape, aff_mat_norm_reshape, select_part_indices, \
            # keypoint_location, aff_prob_gt, masked_fg_aff, valid_aff_gt = mx.sym.Custom(op_type="draw_attention_map",
            #                                                                             im=mx.symbol.BlockGrad(im),
            #                                                                             det_map=d_pred,
            #                                                                             aff_mat=aff_mat_reshape,
            #                                                                             aff_norm=aff_mat_norm_reshape,
            #                                                                             select_part_indices=mx.symbol.BlockGrad(select_part_indices),
            #                                                                             keypoint_location=mx.symbol.BlockGrad(keypoint_location),
            #                                                                             aff_prob_gt=mx.symbol.BlockGrad(aff_prob_gt),
            #                                                                             masked_fg_aff=mx.symbol.BlockGrad(masked_fg_aff),
            #                                                                             valid_aff_gt=mx.symbol.BlockGrad(valid_aff_gt),
            #                                                                             nickname=prefix_name + "_draw_attention_map")

            aff_prob_gt_norm = mx.sym.broadcast_div(aff_prob_gt, mx.sym.maximum(1e-5, aff_prob_gt.sum(axis=-1, keepdims=True)))
            fg_cross_entropy_loss = aff_prob_gt_norm*mx.sym.log(mx.sym.maximum(masked_fg_aff, 1e-5))
            # fg_cross_entropy_loss = mx.sym.Custom(op_type='monitor', data=fg_cross_entropy_loss, nickname='fg_cross_entropy_loss')
            fg_cross_entropy_loss = fg_cross_entropy_loss.sum()

            if cfg.pose.neg_keypoint_post_num > 0:
                # neg_keypoint_loc: [3, N, neg_keypoint_num]
                neg_keypoint_loc = mx.sym.Custom(op_type="neg_keypoint_sampler",
                                                 keypoint_location=mx.sym.BlockGrad(keypoint_location),
                                                 select_part_indices=mx.sym.BlockGrad(select_part_indices),
                                                 aff_mat=mx.sym.BlockGrad(aff_mat_norm_reshape),
                                                 det_score=mx.sym.BlockGrad(d_pred),
                                                 radius = cfg.pose.bg_radius,
                                                 pre_num=cfg.pose.neg_keypoint_pre_num,
                                                 post_num=cfg.pose.neg_keypoint_post_num)

                # bg_aff: [N, neg_keypoint_num, num_part*(K+1)]
                bg_aff = mx.sym.gather_nd(aff_mat_norm_reshape, indices=neg_keypoint_loc, name=prefix_name + "_gather_g_aff")
                # bg_aff_reshape: [N, neg_keypoint_num, num_part, K+1]
                bg_aff_reshape = mx.sym.reshape(bg_aff, shape=(0,0, num_parts, top_k + 1))
                # bg_aff_reshape_self
                bg_aff_reshape_self = mx.sym.slice_axis(bg_aff_reshape, axis=3, begin=top_k, end=top_k+1)

                bg_cross_entropy_loss = mx.sym.log(mx.sym.maximum(bg_aff_reshape_self, 1e-5))
                # bg_cross_entropy_loss = mx.sym.Custom(op_type='monitor', data=bg_cross_entropy_loss, nickname='bg_cross_entropy_loss')
                bg_cross_entropy_loss = bg_cross_entropy_loss.sum()
            else:
                bg_cross_entropy_loss = 0

            valid_count = mx.sym.maximum((aff_prob_gt > 0).sum() + cfg.pose.neg_keypoint_post_num*cfg.TRAIN.BATCH_IMAGES*num_parts, 1)
            # valid_count = mx.sym.Custom(op_type='monitor', data=valid_count, nickname='valid_count')
            aff_loss = mx.sym.MakeLoss(-(fg_cross_entropy_loss+bg_cross_entropy_loss) / valid_count, grad_scale=cfg.pose.aff_loss_weight)

            feature = mx.sym.concat(feature, relation_feat, dim=1)
            feature = mx.sym.Convolution(data=feature, kernel=(1,1), stride=(1,1), num_filter=256, name=prefix_name+"_after_concat_1x1")
            feature = mx.sym.Activation(data=feature, act_type='relu')

            d_pred =  mx.sym.Convolution(data=feature, kernel=(1,1), stride=(1,1), num_filter=num_parts, name=prefix_name+"_det")
            a_pred = mx.sym.Convolution(data=feature, kernel=(1, 1), stride=(1,1), num_filter=num_parts, name=prefix_name + "_association")

            det_preds.append(d_pred)
            association_preds.append(a_pred)
            aff_losses.append(aff_loss)
            aff_labels.append(aff_prob_gt)

        return det_preds, association_preds, aff_losses, aff_labels


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
            im = mx.sym.Variable(name="data")  # img, [N, 3, H ,W]
            data = im
            heatmaps = mx.sym.Variable(name="heatmaps")  # heatmaps of parts, [N, num_parts, H/4, W/4], REMARK 1/4 scale
            masks = mx.sym.Variable(name="masks")  # mask of crowds in coco, [N, H/4, W/4], REMARK 1/4 scale
            # keypoints = mx.sym.Variable(name='keypoints')  # coordinates of keypoints, [N, max_persons, num_parts, 2], REMARK 1/4 scale
            keypoint_visible = mx.sym.Variable(name='keypoint_visible') # [N, max_persons, num_parts]
            keypoint_location  = mx.sym.Variable(name='keypoint_location') # [N, max_person, num_parts, 4]
            keypoint_location = mx.sym.transpose(keypoint_location, axes=(3, 0, 1, 2), name="keypoint_location_transpose")
            # prepare BN func, this one can be easily replaced
            bn = CurrentBN(cfg.network.use_bn_type, 0.9)
        else:
            im = mx.sym.Variable(name="data")  # img, [N, 3, H ,W]
            data = im
            # prepare BN func, this one can be easily replaced
            bn = CurrentBN(cfg.network.use_bn_type, 0.9, use_global_stats=False)
            keypoint_location = None
            keypoint_visible = None
        # pre
        data = conv_sym_wrapper(data=data, prefix="pre1", num_filter=64, kernel=(7, 7), stride=(2, 2), pad=(3, 3), bn=bn, record=self.init_pre_list)
        data = conv_sym_wrapper(data=data, prefix="pre2", num_filter=128, kernel=(3, 3), stride=(1, 1), pad=(1, 1), bn=bn, record=self.init_pre_list)
        data = mx.symbol.Pooling(data=data, kernel=(2, 2), stride=(2, 2), pad=(0, 0), pool_type='max')
        data = conv_sym_wrapper(data=data, prefix="pre3", num_filter=128, kernel=(3, 3), stride=(1, 1), pad=(1, 1), bn=bn, record=self.init_pre_list)
        data = conv_sym_wrapper(data=data, prefix="pre4", num_filter=in_dim, kernel=(3, 3), stride=(1, 1), pad=(1, 1), bn=bn, record=self.init_pre_list)

        # hourglass
        # preds->shape [N, num_stack, C=out_dim, H, W]
        det_preds, association_preds, aff_losses, aff_labels = self.get_stacked_hourglass(data=data,
                                           keypoint_location=keypoint_location, keypoint_visible=keypoint_visible,
                                           num_stack=num_stack, in_dim=in_dim, out_dim=out_dim,
                                           increase_dim=increase_dim, bn=bn, record=self.init_hourglass_list,
                                           num_parts=num_parts, cfg=cfg, is_train=is_train, im=im)

        # calc_loss
        if is_train:
            outside_loss_list = []
            inside_loss_list = []
            det_loss_list = []

            for i in range(len(det_preds)):
                det_loss_list.append(get_det_loss(det_preds[i], heatmaps, masks))

            # stack all stage
            for i in range(len(association_preds)):
                # a_pred: [N, K, H, W, 1]
                a_pred = mx.sym.expand_dims(association_preds[i], axis=4)
                outside_loss, inside_loss = get_inside_outside_loss(feature=a_pred,
                                                                     keypoint_visible=keypoint_visible,
                                                                     keypoint_location=keypoint_location,
                                                                     batch_size=cfg.TRAIN.BATCH_IMAGES,
                                                                     num_keypoint_cls=num_parts,
                                                                     max_persons=max_persons,
                                                                     prefix="stack_{}".format(i))
                outside_loss_list.append(outside_loss * 0.5)
                inside_loss_list.append(inside_loss)

            # stack all stage loss together
            det_loss_all_stage = mx.sym.stack(*det_loss_list).mean()
            outside_loss_all_stage = mx.sym.stack(*outside_loss_list).mean()
            inside_loss_all_stage =  mx.sym.stack(*inside_loss_list).mean()

            outside_loss_all_stage = mx.sym.MakeLoss(outside_loss_all_stage, grad_scale=cfg.pose.outside_loss_weight, name='outside_loss')
            inside_loss_all_stage = mx.sym.MakeLoss(inside_loss_all_stage, grad_scale=cfg.pose.inside_loss_weight, name='inside_loss')
            det_loss_all_stage = mx.sym.MakeLoss(det_loss_all_stage, grad_scale=cfg.pose.det_loss_weight, name='det_loss')

            output_list = [det_loss_all_stage, inside_loss_all_stage, outside_loss_all_stage]

            # get gpu metric
            if cfg.TRAIN.GPU_METRIC:
                for i in range(num_stack + cfg.pose.head_num):
                    output_list.extend(get_detection_loss(det_loss_list[i]))
                    output_list.extend(get_association_loss_inside(inside_loss_list[i]))
                    output_list.extend(get_association_loss_outside(outside_loss_list[i]))

                output_list.extend(get_det_max(mx.symbol.squeeze(mx.sym.slice_axis(data=det_preds[-1], axis=1, begin=3, end=4))))

                if cfg.pose.aff_supervision:
                    for i in range(cfg.pose.head_num):
                        output_list.extend(get_aff_loss(aff_losses[i]))
                        output_list.extend(get_positive_num(aff_labels[i]))

                if cfg.pose.ind_det_loss:
                    for i in range(len(det_preds)):
                        output_list.append(get_postive_det_loss(det_preds[i], heatmaps, masks))
                        output_list.append(get_negative_det_loss(det_preds[i], heatmaps, masks))

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
                if self.cfg.pose.aff_supervision:
                    for i in range(self.cfg.pose.head_num):
                        pred_names.append('Aff_Loss_{}'.format(i))
                        pred_names.append('Aff_GT_Num_{}'.format(i))

                if self.cfg.pose.ind_det_loss:
                    for i in range(4 + self.cfg.pose.head_num):
                        pred_names.append('Pos_DLoss_{}'.format(i))
                        pred_names.append('Neg_DLoss_{}'.format(i))

            return pred_names
        else:
            return

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
        weight_names = ['key_embed', 'val_embed', 'query_embed', 'fusion', 'after_concat_1x1', 'det', 'association', 'sensitve_conv']
        bias_names = ['fusion', 'after_concat_1x1', 'det', 'association','sensitve_conv']
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
