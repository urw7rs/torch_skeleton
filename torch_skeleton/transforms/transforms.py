from . import functions


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class PadBodies:
    def __init__(self, max_bodies, **kwargs):
        self.max_bodies = max_bodies
        self.pad_kwargs = kwargs

    def __call__(self, x):
        return functions.pad_bodies(x, max_bodies=self.max_bodies, **self.pad_kwargs)


class PadFrames:
    def __init__(self, max_frames, axis=1, **kwargs):
        self.max_frames = max_frames
        self.pad_kwargs = kwargs
        self.axis = axis

    def __call__(self, x):
        return functions.pad_frames(
            x, max_frames=self.max_frames, axis=self.axis, **self.pad_kwargs
        )


class SampleFrames:
    def __init__(self, num_frames, num_clips=1, axis=0) -> None:
        self.num_frames = num_frames
        self.num_clips = num_clips
        self.axis = axis

    def __call__(self, x):
        return functions.sample_frames(
            x, num_frames=self.num_frames, num_clips=self.num_clips, axis=self.axis
        )


class SelectKBodies:
    def __init__(self, k):
        self.k = k

    def __call__(self, x):
        return functions.select_k_bodies(x, k=self.k)


class SplitFrames:
    def __call__(self, x):
        return functions.split_frames(x)


class SubJoint:
    def __init__(self, joint_id=1, all=False):
        self.joint_id = joint_id
        self.all = all

    def __call__(self, x):
        return functions.sub_center_joint(x, joint_id=self.joint_id, all=self.all)


class ParallelBone:
    def __init__(self, first_id, second_id, axis=2) -> None:
        super().__init__()
        self.first_id = first_id
        self.second_id = second_id
        self.axis = axis

    def __call__(self, x):
        return functions.parallel_bone(
            x, first_id=self.first_id, second_id=self.second_id, axis=self.axis
        )


class RandomShift:
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, x):
        return functions.random_shift(x, low=self.low, high=self.high)


class RandomRotate:
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, x):
        return functions.random_rotate(x, self.degrees)


class Denoise:
    def __call__(self, x):
        return functions.get_raw_denoised_data(x)


class SortByMotion:
    def __call__(self, x):
        return functions.select_k_bodies(x, k=x.shape[0])


class DenoiseByLength:
    def __call__(self, x):
        return functions.denoising_by_length(x)


class DenoiseBySpread:
    def __call__(self, x):
        return functions.denoising_by_spread(x)


class DenoiseByMotion:
    def __call__(self, x):
        return functions.denoising_by_motion(x)


class MergeBodies:
    def __call__(self, x):
        return functions.merge_bodies(x)


class NonZeroFrames:
    def __call__(self, x):
        return functions.nonzero_frames(x)
