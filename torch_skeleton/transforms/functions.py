import numpy as np

import einops


def get_mask(x):
    zeros = np.zeros_like(x)
    zeros[x.sum(axis=(2, 3)) != 0] = 1
    return zeros


def get_indices(x):
    indices = np.where(x.sum(axis=(2, 3)) != 0)
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
