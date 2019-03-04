import mxnet as mx

def get_inside_outside_loss(feature, keypoint_visible, keypoint_location, batch_size, max_persons,
                            num_keypoint_cls, prefix):
    # visible_keypoint_num: [N, P, 1]
    visible_keypoint_num = keypoint_visible.sum(axis=2, keepdims=True)
    # visible_person: [N, P, 1]
    visible_person = visible_keypoint_num > 0
    # visible_person_t: [N, 1, P]
    visible_person_t = mx.sym.transpose(visible_person, axes=(0, 2, 1))
    # visible_person_pair: [N, P, P]
    eye_mat = mx.sym.eye(max_persons).reshape(shape=(1, max_persons, max_persons))
    visible_person_pair = mx.sym.broadcast_mul(mx.sym.broadcast_mul(visible_person, visible_person_t), 1 - eye_mat)
    # visible_person_num: [N, 1, 1]
    visible_person_num = visible_person.sum(axis=1)

    # feature: [N, K, H, W, C]
    keypoint_feats = mx.sym.gather_nd(feature, keypoint_location)

    # keypoint_feat: [N, P, K, C]
    keypoint_feats = mx.sym.reshape(keypoint_feats, shape=(batch_size, -1, num_keypoint_cls, 0),
                                    name=prefix + "_keypoint_feats_reshape")

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
    mean_sqr_diff = mx.sym.broadcast_sub(mean_keypoint_feats, mean_keypoint_feats_t,
                                         name=prefix + '_braodcast_sub_mean_sqr_diff')
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
    instance_sqr_diff = mx.sym.broadcast_sub(keypoint_feats, mean_keypoint_feats,
                                             name=prefix + '_broadcast_sub_instance_sqr_diff')
    instance_sqr_diff = mx.sym.square(instance_sqr_diff).sum(axis=3)

    instance_sqr_diff = instance_sqr_diff * keypoint_visible

    # inside loss
    inside_loss = instance_sqr_diff.sum(axis=2, keepdims=True) / mx.sym.maximum(1, visible_keypoint_num)
    inside_loss = inside_loss.sum(axis=1) / mx.sym.maximum(1, visible_person_num)

    outside_loss_mean = mx.sym.mean(outside_loss, name="outside_loss_mean")
    inside_loss_mean = mx.sym.mean(inside_loss, name="inside_loss_mean")

    return outside_loss_mean, inside_loss_mean


def get_aggregate_keypoint_loc(keypoint_location):
    keypoint_location_batch = mx.sym.slice_axis(keypoint_location, axis=0, begin=0, end=1)
    keypoint_location_position = mx.sym.slice_axis(keypoint_location, axis=0, begin=2, end=4)
    keypoint_location_t = mx.sym.concat(keypoint_location_batch, keypoint_location_position, dim=0)

    keypoint_location_t = mx.sym.reshape(keypoint_location_t, shape=(0, 0, -1))
    return keypoint_location_t


def get_det_loss(det_pred, heatmaps, masks):
    det_loss = mx.symbol.square(data=(det_pred - heatmaps))
    masks_4d = mx.symbol.expand_dims(masks, axis=1)
    det_loss = mx.symbol.broadcast_mul(det_loss, masks_4d).mean()

    return det_loss
