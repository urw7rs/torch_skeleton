import os.path as osp

import numpy as np

import einops


def get_mask(x):
    zeros = np.zeros_like(x)
    zeros[x.sum(axis=(2, 3)) != 0] = 1
    return zeros


def get_indices(x, axis=(2, 3)):
    indices = np.where(x.sum(axis=axis) != 0)
    return indices


def _pad(x, padding, axis, **pad_kwargs):
    shape = [(0, 0)] * x.ndim
    shape[axis] = padding

    return np.pad(x, shape, **pad_kwargs)


def pad_bodies(x, max_bodies, axis=0, **pad_kwargs):
    diff = max_bodies - x.shape[axis]
    if diff > 0:
        padding = (diff // 2, diff - diff // 2)
        x = _pad(x, padding, axis=axis, **pad_kwargs)
    return x


def pad_frames(x, max_frames, axis=1, **pad_kwargs):
    diff = max_frames - x.shape[axis]
    if diff > 0:
        padding = (diff // 2, diff - diff // 2)
        x = _pad(x, padding, axis=axis, **pad_kwargs)
    return x


def unpad_bodies(x, axis=0):
    indices = x.nonzero()

    index = np.unique(indices[axis])
    x = x.take(index, axis=axis)
    return x


def unpad_frames(x, axis=1):
    indices = x.nonzero()

    index = indices[axis]
    start = index.min()
    end = index.max()

    x = x.take(np.arange(start, end + 1), axis=axis)
    return x


def pad_previous(x):
    for i_p, person in enumerate(x):
        if person.sum() == 0:
            continue
        if person[0].sum() == 0:
            index = person.sum(-1).sum(-1) != 0
            tmp = person[index].copy()
            person *= 0
            person[: len(tmp)] = tmp
        for i_f, frame in enumerate(person):
            if frame.sum() == 0:
                if person[i_f:].sum() == 0:
                    rest = len(person) - i_f
                    num = int(np.ceil(rest / i_f))
                    pad = np.concatenate([person[0:i_f] for _ in range(num)], 0)[:rest]
                    x[i_p, i_f:] = pad
                    break
    return x


def unit_vector(v):
    assert (v**2).sum() != 0

    axis = np.ndim(v) - 1
    return v / np.linalg.norm(v, axis=axis)


def angle_between(v1, v2):
    if np.abs(v1).sum() < 1e-6 or np.abs(v2).sum() < 1e-6:
        return 0

    v1 = unit_vector(v1)
    v2 = unit_vector(v2)

    v1_shape = v1.shape
    v2_shape = v2.shape

    dot_product = np.reshape(
        np.einsum(
            "bij,bji->bi",
            np.reshape(v1, (-1, 1, v1_shape[-1])),
            np.reshape(v2, (-1, v2_shape[-1], 1)),
        ),
        v1_shape[:-1],
    )

    return np.arccos(dot_product)


def calc_rot_matrix(theta, axis):
    cos = np.cos(theta)
    sin = np.sin(theta)

    assert axis in [0, 1, 2]

    if axis == 0:
        R = np.array(
            [
                [1, 0, 0],
                [0, cos, -sin],
                [0, sin, cos],
            ]
        )
    elif axis == 1:
        R = np.array(
            [
                [cos, 0, sin],
                [0, 1, 0],
                [-sin, 0, cos],
            ]
        )
    elif axis == 2:
        R = np.array(
            [
                [cos, -sin, 0],
                [sin, cos, 0],
                [0, 0, 1],
            ]
        )

    return R


def sample_frames(x, num_frames, num_clips, axis=0):
    if num_clips > 1:
        clips = [_sample_frames(x, num_frames=num_frames) for _ in range(num_clips)]

        x = np.stack(clips, axis=axis)
    else:
        x = _sample_frames(x, num_frames=num_frames)

    return x


def _sample_frames(x, num_frames):
    length = x.shape[1]
    if length <= num_frames:
        return x

    chunk_size = length // num_frames
    last_chunk_size = length % num_frames

    if last_chunk_size > 0:
        start_offset = np.random.randint(last_chunk_size)
    else:
        start_offset = 0

    start = np.arange(num_frames) * chunk_size
    chunk_offsets = np.random.randint(chunk_size, size=num_frames)

    x = x[:, start + start_offset + chunk_offsets]

    return x


def sample_frames1(x, num_frames):
    length = x.shape[1]
    if length <= num_frames:
        return x

    splits = np.array_split(x, num_frames, axis=1)

    sampled_frames = []
    for split in splits:
        num_frames_of_split = split.shape[1]
        if num_frames_of_split > 1:
            num_frames_of_split -= 1
        index = np.random.randint(low=0, high=num_frames_of_split)
        sampled_frames.append(split[:, index : index + 1])

    sampled_frames = np.concatenate(sampled_frames, axis=1)
    return sampled_frames


def select_k_bodies(x, k):
    num_bodies = x.shape[0]

    if num_bodies > k:
        std_summed = []
        for m in range(num_bodies):
            body = x[m]
            joint_sum = einops.reduce(body, "t v c -> t", "sum")
            nonzero = body[joint_sum != 0]

            std = einops.reduce(nonzero, "t v c -> c", np.std)
            sum_of_std = einops.reduce(std, "c -> 1", "sum")

            std_summed.append(sum_of_std)

        motion = np.concatenate(std_summed, axis=0)
        indices = np.argpartition(motion, -k)[: num_bodies - k : -1]

        x = x[indices]

    return x


def split_frames(x):
    num_bodies, num_frames, J, C = x.shape

    if num_bodies == 1:
        return x

    x = np.transpose(x, axes=(1, 0, 2, 3))
    x = np.reshape(x, newshape=(num_bodies * num_frames, J, C))

    x = x[x.sum(axis=(1, 2)) != 0]

    return np.expand_dims(x, axis=0)


def sub_center_joint(x, joint_id, all=False):
    mask = get_mask(x)

    if all:
        x -= x[:, :, joint_id : joint_id + 1] * mask
    else:
        x -= x[0, 0:1, joint_id : joint_id + 1] * mask

    return x


def parallel_bone(x, first_id=0, second_id=1, axis=2):
    indices = get_indices(x)
    joints = x[indices]

    first_joint = joints[0, first_id]
    second_joint = joints[0, second_id]

    bone = second_joint - first_joint

    align_axis = np.zeros_like(bone)
    align_axis[..., axis] = 1

    theta = angle_between(bone, align_axis)

    if theta == 0:
        return x

    if theta > np.pi:
        theta = np.pi - theta
        theta = -theta

    rot_axis = np.cross(bone, align_axis)
    mrp = unit_vector(rot_axis) * np.tan(theta / 4)
    r = Rotation.from_mrp(mrp)
    R = r.as_matrix()

    joints = np.einsum("ij,bvj->bvi", R, joints)

    x[indices] = joints

    return x


def random_shift(x, low, high):
    M, T, V, C = x.shape
    offset = np.random.uniform(low=low, high=high, size=(1, 1, 1, C))
    return x + offset


def random_rotate(x, theta):
    theta_x, theta_y, theta_z = np.deg2rad(
        np.random.uniform(low=-theta, high=theta, size=(3,))
    )

    R_x = calc_rot_matrix(theta_x, axis=0)
    R_y = calc_rot_matrix(theta_y, axis=1)
    R_z = calc_rot_matrix(theta_z, axis=2)

    R = np.matmul(R_x, R_y)
    R = np.matmul(R, R_z)

    indices = get_indices(x)
    joints = x[indices]
    x[indices] = np.einsum("ij,bvj->bvi", R, joints)
    return x


missing_count = 0
noise_len_thres = 11
noise_spr_thres1 = 0.8
noise_spr_thres2 = 0.69754
noise_mot_thres_lo = 0.089925
noise_mot_thres_hi = 2


def denoising_by_length(x):
    """
    Denoising data based on the frame length for each bodyID.
    Filter out the bodyID which length is less or equal than the predefined threshold.
    """
    filtered_bodies = []
    for body in np.split(x, indices_or_sections=x.shape[0], axis=0):
        _, indices = get_indices(body)
        length = indices.max() - indices.min() + 1
        if length > noise_len_thres:
            filtered_bodies.append(body)

    x = np.concatenate(filtered_bodies, axis=0)
    return x


def get_valid_frames_by_spread(points):
    """
    Find the valid (or reasonable) frames (index) based on the spread of X and Y.
    :param points: joints or colors
    """
    num_frames = points.shape[0]
    valid_frames = []
    for i in range(num_frames):
        x = points[i, :, 0]
        y = points[i, :, 1]
        if (x.max() - x.min()) <= noise_spr_thres1 * (y.max() - y.min()):  # 0.8
            valid_frames.append(i)
    return np.array(valid_frames)


def denoising_by_spread(x):
    """
    Denoising data based on the spread of Y value and X value.
    Filter out the bodyID which the ratio of noisy frames is higher than the predefined
    threshold.
    bodies_data: contains at least 2 bodyIDs
    """
    denoised_bodies = []
    for body in np.split(x, indices_or_sections=x.shape[0], axis=0):
        valid_indices = get_valid_frames_by_spread(body[0])
        _, indices = get_indices(body)
        num_frames = indices.shape[0]
        num_noise = num_frames - valid_indices.shape[0]
        if num_noise == 0:
            denoised_bodies.append(body)
            continue

        ratio = num_noise / float(num_frames)
        if ratio < noise_spr_thres2:  # 0.69754
            denoised_bodies.append(body)
            # TODO: Consider removing noisy frames for each bodyID

    x = np.concatenate(denoised_bodies, axis=0)
    return x


def denoising_by_motion(x):
    """
    Filter out the bodyID which motion is out of the range of predefined interval
    """
    denoised_bodies = [x[0:1]]
    for body in np.split(x[1:], indices_or_sections=x.shape[0] - 1, axis=0):
        motion = np.sum(np.var(body.reshape(-1, 3), axis=0))

        valid_indices = get_valid_frames_by_spread(body[0])
        valid_body = body[:, valid_indices]
        denoised_motion = np.sum(np.var(valid_body.reshape(-1, 3), axis=0))

        motion = min(motion, denoised_motion)
        if (motion < noise_mot_thres_lo) or (motion > noise_mot_thres_hi):
            pass
        else:
            denoised_bodies.append(body)

    x = np.concatenate(denoised_bodies, axis=0)
    return x


def denoising_bodies_data(x):
    """
    Denoising data based on some heuristic methods, not necessarily correct for all samples.
    Return:
      denoised_bodies_data (list): tuple: (bodyID, body_data).
    """

    # Step 1: Denoising based on frame length.
    x = denoising_by_length(x)

    if x.shape[0] == 1:  # only has one bodyID left after step 1
        return x

    # Step 2: Denoising based on spread.
    x = denoising_by_spread(x)

    if x.shape[0] == 1:
        return x

    x = denoising_by_motion(x)
    return x


def get_one_actor_points(x):
    return x[0:1]


def intersect_indices(actor1, actor2):
    _, t_indices1 = get_indices(actor1)
    _, t_indices2 = get_indices(actor2)

    return np.intersect1d(t_indices1, t_indices2)


def get_two_actors_points(x):
    """
    Get the first and second actor's joints positions and colors locations.
    # Arguments:
        bodies_data (dict): 3 key-value pairs: 'name', 'data', 'num_frames'.
        bodies_data['data'] is also a dict, while the key is bodyID, the value is
        the corresponding body_data which is also a dict with 4 keys:
          - joints: raw 3D joints positions. Shape: (num_frames x 25, 3)
          - colors: raw 2D color locations. Shape: (num_frames, 25, 2)
          - interval: a list which records the frame indices.
          - motion: motion amount
    # Return:
        joints, colors.
    """
    x = denoising_bodies_data(x)  # Denoising data

    num_bodies = x.shape[0]

    if num_bodies > 1:  # Only left one actor after denoising
        main_actor = x[0:1]
        second_actor = np.zeros_like(main_actor)

        actors = np.split(x[1:], indices_or_sections=num_bodies - 1, axis=0)

        for actor in actors:
            _, t_indices1 = get_indices(main_actor)
            _, t_indices2 = get_indices(actor)

            intersect = np.intersect1d(t_indices1, t_indices2)

            if len(intersect) == 0:  # no overlap with actor1
                main_actor[:, t_indices2] = actor[:, t_indices2]
            else:
                _, t_indices1 = get_indices(second_actor)
                _, t_indices2 = get_indices(actor)

                intersect = np.intersect1d(t_indices1, t_indices2)
                if len(intersect) == 0:
                    second_actor[:, t_indices2] = actor[:, t_indices2]

        x = np.concatenate([main_actor, second_actor], axis=0)

    return x


def get_raw_denoised_data(x):
    """
    Get denoised data (joints positions and color locations) from raw skeleton sequences.
    For each frame of a skeleton sequence, an actor's 3D positions of 25 joints represented
    by an 2D array (shape: 25 x 3) is reshaped into a 75-dim vector by concatenating each
    3-dim (x, y, z) coordinates along the row dimension in joint order. Each frame contains
    two actor's joints positions constituting a 150-dim vector. If there is only one actor,
    then the last 75 values are filled with zeros. Otherwise, select the main actor and the
    second actor based on the motion amount. Each 150-dim vector as a row vector is put into
    a 2D numpy array where the number of rows equals the number of valid frames. All such
    2D arrays are put into a list and finally the list is serialized into a cPickle file.
    For the skeleton sequence which contains two or more actors (mostly corresponds to the
    last 11 classes), the filename and actors' information are recorded into log files.
    For better understanding, also generate RGB+skeleton videos for visualization.
    """

    num_bodies, num_frames, num_joints, _ = x.shape

    x = select_k_bodies(x, k=x.shape[0])

    if num_bodies == 1:  # only 1 actor
        x = get_one_actor_points(x)
    else:  # more than 1 actor, select two main actors
        x = get_two_actors_points(x)

        x = unpad_frames(x)

    return x
