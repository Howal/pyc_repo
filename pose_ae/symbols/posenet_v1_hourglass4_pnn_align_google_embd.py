import mxnet as mx
import numpy as np
from common.lib.utils.symbol import Symbol
from common.backbone.hourglass_v1 import hourglass_v1, conv_sym_wrapper, CurrentBN
from common.gpu_metric import *
from common.operator_py.monitor_op import *
from common.operator_py.monitor_op_multivar import *
from relation_helper_sym import *

class posenet_v1_hourglass4_pnn_align_google_embd(Symbol):
    def __init__(self, FP16=False):
        """
        Use __init__ to define parameter network needs
        """
        # FP16 is not used for now
        self.FP16 = FP16
        self.init_pre_list = []
        self.init_hourglass_list = []
        self.cfg = None

    def get_stacked_hourglass(self, data, num_stack=4, in_dim=256, out_dim=68, increase_dim=128,
                              bn=CurrentBN(False, 0.9), record=[], num_parts=17, cfg=None, is_train=False, im=None):

        det_preds = []
        association_preds = []
        features = []
        for i in range(num_stack):
            body = hourglass_v1(data=data, num_stage=4, in_dim=in_dim, increase_dim=increase_dim, bn=bn, prefix="hg{}".format(i + 1), record=record)
            body = conv_sym_wrapper(data=body, prefix="hg{}_out1".format(i + 1), num_filter=in_dim, kernel=(3, 3), stride=(1, 1), pad=(1, 1), bn=bn, record=record)
            feature = conv_sym_wrapper(data=body, prefix="hg{}_out2".format(i + 1), num_filter=in_dim, kernel=(3, 3), stride=(1, 1), pad=(1, 1), bn=bn, record=record)

            if i < num_stack - 1 or cfg.pose.bottom_up_loss:
                out = conv_sym_wrapper(data=feature, prefix="hg{}_out3".format(i + 1), num_filter=out_dim, kernel=(1, 1), stride=(1, 1), pad=(0, 0), bn=bn, relu=False, record=record)
                d_pred = mx.sym.slice_axis(data=out, axis=1, begin=0, end=num_parts)  # shape, [N, num_parts, H, W]
                a_pred = mx.sym.slice_axis(data=out, axis=1, begin=num_parts, end=2*num_parts)  # shape, [N, num_parts, H, W]

                det_preds.append(d_pred)
                association_preds.append(a_pred)

            features.append(feature)
            if i != num_stack - 1:
                data_preds = conv_sym_wrapper(data=out, prefix="hg{}_merge_preds".format(i + 1), num_filter=in_dim, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                              bn=bn, relu=False, record=record)
                data_feats = conv_sym_wrapper(data=feature, prefix="hg{}_merge_feats".format(i + 1), num_filter=in_dim, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                              bn=bn, relu=False, record=record)
                data = data + data_preds + data_feats

        return det_preds, association_preds, features

    def aggregate_loc(self, proposal_location):
        proposal_location_batch = mx.sym.slice_axis(proposal_location, axis=0, begin=0, end=1)
        proposal_location_y = mx.sym.slice_axis(proposal_location, axis=0, begin=2, end=3)
        proposal_location_x = mx.sym.slice_axis(proposal_location, axis=0, begin=3, end=4)

        proposal_location_t = mx.sym.concat(proposal_location_batch, proposal_location_y, proposal_location_x, dim=0)
        return proposal_location_t


    def get_google_embed(self, proposal_location, wave_length, embed_dim, dif_mul):
        embed_dim = embed_dim/2
        # proposal_location: [N, max_proposals, num_parts, 4]
        # part_label: [N, max_proposals, num_parts]
        # loc:[N, max_proposals, num_parts, 2]
        # proposal_location = mx.sym.Custom(op_type='monitor', data=proposal_location, nickname='proposal_location')

        loc = mx.sym.slice_axis(proposal_location, axis=3, begin=2, end=4)

        # loc = mx.sym.Custom(op_type='monitor', data=loc, nickname='loc')
        # center_loc: [N, max_proposals, 1, 2]
        center_loc = loc.mean(axis=2, keepdims=True)
        # center_loc = mx.sym.Custom(op_type='monitor', data=center_loc, nickname='center_loc')
        # diff_loc: [N, max_poroposals, num_parts, 2]
        diff_loc = mx.sym.broadcast_sub(loc, center_loc)
        # diff_loc = mx.sym.Custom(op_type='monitor', data=diff_loc, nickname='diff_loc')

        # norm_base: [N, max_proposals, num_parts, 1]
        norm_base = mx.sym.sqrt(mx.sym.square(diff_loc).sum(axis=3, keepdims=True))
        # norm_base = mx.sym.Custom(op_type='monitor', data=norm_base, nickname='norm_base')
        # norm_base: [N, max_proposals, 1, 1]
        norm_base = norm_base.mean(axis=2, keepdims=True)
        # norm_base = mx.sym.Custom(op_type='monitor', data=norm_base, nickname='norm_base')
        norm_base = mx.sym.maximum(1e-5, norm_base)
        normed_diff_loc = mx.sym.broadcast_div(diff_loc, norm_base)
        # normed_diff_loc = mx.sym.Custom(op_type='monitor', data=normed_diff_loc, nickname='normed_diff_loc')

        # base: [embed_dim/2]
        base = mx.sym.pow(wave_length, mx.sym.arange(0, embed_dim) / embed_dim)
        # base: [1,1,1,1,embed_dim/2]
        base = base.reshape(shape=(1,1,1,1,embed_dim))

        # div_mat: [N, max_poroposals, num_parts, 2, embed_dim/2]
        div_mat = mx.sym.broadcast_div(dif_mul * normed_diff_loc.expand_dims(axis=4), base)
        sin_mat = mx.sym.sin(div_mat)
        cos_mat = mx.sym.cos(div_mat)
        # sin_mat = mx.sym.Custom(op_type='monitor', data=sin_mat, nickname='sin_mat')

        # embed: [N, max_poroposals, num_parts, 2, embed_dim]
        embed_feat = mx.sym.concat(sin_mat, cos_mat, dim=4)
        embed_feat = embed_feat.reshape(shape=(0,0,0,-1))

        return embed_feat


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
            keypoint_visible = mx.sym.Variable(name='keypoint_visible') # [N, max_persons, num_parts]
            keypoint_location  = mx.sym.Variable(name='keypoint_location') # [N, max_person, num_parts, 4]
            keypoint_location = mx.sym.transpose(keypoint_location, axes=(3, 0, 1, 2), name="keypoint_location_transpose")

            proposal_location = mx.sym.Variable(name='proposal_location') # proposal_label: [N, max_proposals, num_parts, 4]
            proposal_label = mx.sym.Variable(name='proposal_label') # proposal_label: [N, max_proposals]
            part_label = mx.sym.Variable(name='part_label') # part_label: [N, max_proposals, num_parts]

            if cfg.pose.use_offset:
                gt_offset = mx.sym.Variable(name='offset') # gt_offset: [N, max_proposals, num_parts, 2]


            # prepare BN func, this one can be easily replaced
            bn = CurrentBN(cfg.network.use_bn_type, 0.9)
        else:
            im = mx.sym.Variable(name="data")  # img, [N, 3, H ,W]
            data = im
            # prepare BN func, this one can be easily replaced
            bn = CurrentBN(cfg.network.use_bn_type, 0.9, use_global_stats=False)
            keypoint_location = None
            keypoint_visible = None
            proposal_location = mx.sym.Variable(name='proposal_location')  # proposal_label: [N, max_proposals, num_parts, 4]

        # pre
        data = conv_sym_wrapper(data=data, prefix="pre1", num_filter=64, kernel=(7, 7), stride=(2, 2), pad=(3, 3), bn=bn, record=self.init_pre_list)
        data = conv_sym_wrapper(data=data, prefix="pre2", num_filter=128, kernel=(3, 3), stride=(1, 1), pad=(1, 1), bn=bn, record=self.init_pre_list)
        data = mx.symbol.Pooling(data=data, kernel=(2, 2), stride=(2, 2), pad=(0, 0), pool_type='max')
        data = conv_sym_wrapper(data=data, prefix="pre3", num_filter=128, kernel=(3, 3), stride=(1, 1), pad=(1, 1), bn=bn, record=self.init_pre_list)
        data = conv_sym_wrapper(data=data, prefix="pre4", num_filter=in_dim, kernel=(3, 3), stride=(1, 1), pad=(1, 1), bn=bn, record=self.init_pre_list)

        # hourglass
        # preds->shape [N, num_stack, C=out_dim, H, W]
        det_preds, association_preds, features = self.get_stacked_hourglass(data=data,
                                           num_stack=num_stack, in_dim=in_dim, out_dim=out_dim,
                                           increase_dim=increase_dim, bn=bn, record=self.init_hourglass_list,
                                           num_parts=num_parts, cfg=cfg, is_train=is_train, im=im)
        # PNN
        proposal_location_reshape = proposal_location.transpose(axes=(3, 0, 1, 2)) # proposal_location: [4, N, max_proposals, num_parts]
        proposal_location_reshape = self.aggregate_loc(proposal_location_reshape) #proposal_location_reshape: [3, N, max_proposals, num_parts]
        proposal_location_reshape = proposal_location_reshape.transpose(axes=(1,2,3,0)) #proposal_location_reshape: [N, max_proposals, num-parts, 3]
        proposal_location_reshape = proposal_location_reshape.reshape((-1, 3)) #[N * max_proposals * num_parts, 3]

        # proposal_location_reshape = mx.sym.Custom(op_type='monitor', data=proposal_location_reshape, nickname='proposal_location_reshape')
        PNN_feature = mx.sym.contrib.FeatureSampling(data=features[3], coord=proposal_location_reshape) # PNN_feature: [N * max_proposals * num_parts, 256]

        # google_embd: [N, max_proposals, num_parts, 2*embed_dim]
        google_embd = self.get_google_embed(proposal_location, wave_length=cfg.pose.google_embed_wave_length, embed_dim=cfg.pose.google_embed_dim, dif_mul=cfg.pose.google_embed_diff_mul)
        # google_embd = mx.sym.Custom(op_type='monitor', data=google_embd, nickname="google_embd")

        google_embd_reshape = google_embd.reshape(shape=(-1, num_parts*cfg.pose.google_embed_dim*2)) #google_embd: [N*max_proposals, num_parts*2*embed_dim]
        google_embd_fc = mx.sym.FullyConnected(data=google_embd_reshape, num_hidden=cfg.pose.google_embed_fc_dim, name='google_embd_fc')
        # google_embd_fc = mx.sym.Custom(op_type='monitor', data=google_embd_fc, nickname="google_embd_fc")

        # PNN_feature: [N * max_proposals, 17 * 256]
        PNN_feature = PNN_feature.reshape(shape=(-1, num_parts * 256))
        concated_pnn_feature = mx.sym.concat(PNN_feature, google_embd_fc, name='feature_concat', dim=1)
        # concated_pnn_feature = mx.sym.Custom(op_type='monitor', data=concated_pnn_feature, nickname="concated_pnn_feature")

        pnn_fc1 = mx.sym.FullyConnected(data=concated_pnn_feature, num_hidden=1024, name="pnn_fc1")
        pnn_fc1_relu = mx.sym.Activation(data=pnn_fc1, act_type='relu', name="pnn_fc1_relu")
        pnn_fc2 = mx.sym.FullyConnected(data=pnn_fc1_relu, num_hidden=1024, name="pnn_fc2")
        pnn_fc2_relu = mx.sym.Activation(data=pnn_fc2, act_type='relu', name="pnn_fc2_relu")

        # pose_cls: [N*max_proposals, 2]
        pose_cls = mx.sym.FullyConnected(data=pnn_fc2_relu, num_hidden=2, name="pose_cls")
        # pose_cls: [N*max_proposals, 17 * 2]
        part_cls = mx.sym.FullyConnected(data=pnn_fc2_relu, num_hidden=num_parts*2, name="part_cls")
        # part_cls_reshape : [N, max_proposals, 17, 2]
        part_cls_reshape = mx.sym.Reshape(part_cls, shape=(-1,cfg.pose.max_proposals, num_parts, 2))
        # part_cls_reshape: [N, 2, max_proposals, 17]
        part_cls_reshape = mx.sym.transpose(part_cls_reshape, axes=(0, 3, 1, 2))

        if cfg.pose.use_offset:
            pred_offset = mx.sym.FullyConnected(data=pnn_fc2_relu, num_hidden=num_parts * 2, name='pred_offset')
            pred_offset_reshape = mx.sym.reshape(pred_offset, shape=(-1, cfg.pose.max_proposals, num_parts, 2), name="pred_offset_reshape")

        output_list = []
        # calc_loss
        if is_train:
            proposal_label_reshape = proposal_label.reshape((-1))

            # pnn loss:
            part_cls_prob = mx.sym.SoftmaxOutput(data=part_cls_reshape, label=part_label,
                                                 normalization='valid', use_ignore=True, multi_output=True, ignore_label=-1, name='part_cls_prob',
                                                 grad_scale=cfg.pose.part_cls_loss_weight)
            pose_cls_prob = mx.sym.SoftmaxOutput(data=pose_cls, label=proposal_label_reshape,
                                                 normalization='valid', use_ignore=True, ignore_label=-1, name='pose_cls_prob',
                                                 grad_scale=cfg.pose.pose_cls_loss_weight)


            part_label_reshape = part_label.reshape((-1))
            part_cls_prob_reshape = part_cls_prob.transpose(axes=(0,2,3,1)).reshape((-1, 2))

            # pose metric
            output_list.extend(get_log_loss(pose_cls_prob, proposal_label_reshape))
            output_list.extend(get_acc(pose_cls_prob, proposal_label_reshape))
            output_list.extend(get_acc(pose_cls_prob, proposal_label_reshape, cls=1))
            output_list.extend(get_acc(pose_cls_prob, proposal_label_reshape, cls=0))

            # part metric
            output_list.extend(get_log_loss(part_cls_prob_reshape, part_label_reshape))
            output_list.extend(get_acc(part_cls_prob_reshape, part_label_reshape))
            output_list.extend(get_acc(part_cls_prob_reshape, part_label_reshape, cls=1))
            output_list.extend(get_acc(part_cls_prob_reshape, part_label_reshape, cls=0))

            # FG ratio
            output_list.extend(get_fg_ratio(proposal_label))
            # Num of proposals
            output_list.extend(get_num_proposal(proposal_label, self.cfg.TRAIN.BATCH_IMAGES))

            if cfg.pose.use_offset:
                offset_weight = part_label == 1
                offset_loss = offset_weight * mx.sym.smooth_l1(scalar=3.0, data=(pred_offset_reshape - gt_offset)).sum(axis=-1)
                offset_loss = offset_loss.sum() / offset_weight.sum()
                offset_loss = mx.sym.MakeLoss(data=offset_loss, grad_scale=cfg.pose.offset_loss_weight, name='offset_loss')
                output_list.extend(get_offset_loss(offset_loss))


            if cfg.pose.bottom_up_loss:
                # other loss:
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


                output_list.extend([det_loss_all_stage, inside_loss_all_stage, outside_loss_all_stage])

                # get gpu metric
                if cfg.TRAIN.GPU_METRIC:
                    for i in range(num_stack):
                        output_list.extend(get_detection_loss(det_loss_list[i]))
                        output_list.extend(get_association_loss_inside(inside_loss_list[i]))
                        output_list.extend(get_association_loss_outside(outside_loss_list[i]))

                    output_list.extend(get_det_max(mx.symbol.squeeze(mx.sym.slice_axis(data=det_preds[-1], axis=1, begin=3, end=4))))

                else:
                    raise ValueError('No CPU metric is supported now!')

            group = mx.sym.Group(output_list)
        else:
            part_cls_prob = mx.sym.SoftmaxActivation(data=part_cls_reshape, name='part_cls_prob', mode='channel')
            pose_cls_prob = mx.sym.SoftmaxActivation(data=pose_cls, name='pose_cls_prob')

            output_list = [pose_cls_prob, part_cls_prob, proposal_location]
            if cfg.pose.use_offset:
                output_list.extend([pred_offset_reshape])

            group = mx.sym.Group(output_list)

        self.sym = group
        return group

    def get_pred_names(self, is_train, gpu_metric=False):
        if is_train:
            pred_names = []
            pred_names.extend(['pose_logloss', 'pose_logloss_inst_num'])
            pred_names.extend(['pose_acc', 'pose_acc_inst_num'])
            pred_names.extend(['pose_fg_acc', 'pose_acc_fg_inst_num'])
            pred_names.extend(['pose_bg_acc', 'pose_acc_bg_inst_num'])
            pred_names.extend(['part_logloss', 'part_logloss_inst_num'])
            pred_names.extend(['part_acc', 'pose_acc_inst_num'])
            pred_names.extend(['part_fg_acc', 'pose_fg_acc_inst_num'])
            pred_names.extend(['part_bg_acc', 'pose_bg_acc_inst_num'])
            pred_names.extend(['fg_ratio'])
            pred_names.extend(['num_proposal'])

            if self.cfg.pose.use_offset:
                pred_names.extend(['offset_loss'])

            if self.cfg.pose.bottom_up_loss:
                pred_names.extend(['d_loss', 'a_loss_inside', 'a_loss_outside'])
                if gpu_metric:
                    for i in range(4):
                        pred_names.append('D_Loss_{}'.format(i))
                        pred_names.append('A_Loss_Inside_{}'.format(i))
                        pred_names.append('A_Loss_Outside_{}'.format(i))

                    pred_names.append('Det_Max')
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

    def init_weight_pnn(self, cfg, arg_params, aux_params):
        weight_names = ['pnn_fc1', 'pnn_fc2', 'pose_cls', 'part_cls', 'google_embd_fc']
        bias_names = ['pnn_fc1', 'pnn_fc2', 'pose_cls', 'part_cls', 'google_embd_fc']

        if cfg.pose.use_offset:
            weight_names.extend(['pred_offset'])
            bias_names.extend(['pred_offset'])

        for weight_name in weight_names:
            weight_name = weight_name + '_weight'
            weight_shape = self.arg_shape_dict[weight_name]

            if len(weight_shape) == 4:
                # Conv Weight
                fan_in = float(weight_shape[1]) * weight_shape[2] * weight_shape[3]
            else:
                # Fc Weight
                fan_in = float(weight_shape[1])

            bound = np.sqrt(6 / ((1 + 5) * fan_in))

            if cfg.pose.param_init == 'normal':
                arg_params[weight_name] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict[weight_name])
            else:
                arg_params[weight_name] = mx.random.uniform(-bound, bound, shape=weight_shape)

        for bias_name in bias_names:
            bias_name = bias_name + '_bias'
            if cfg.pose.param_init == 'normal':
                arg_params[bias_name] = mx.nd.zeros(shape=self.arg_shape_dict[bias_name])
            else:
                arg_params[bias_name] = mx.random.uniform(-bound, bound, self.arg_shape_dict[bias_name])

    def init_weight(self, cfg, arg_params, aux_params):
        self.init_weight_pre(cfg, arg_params, aux_params)
        self.init_weight_hourglass(cfg, arg_params, aux_params)
        self.init_weight_pnn(cfg, arg_params, aux_params)