from . import functional as F

__all__ = [
    "Compose",
    "PadBodies",
    "PadFrames",
    "SampleFrames",
    "SelectKBodies",
    "SplitFrames",
    "CenterJoint",
    "ParallelBone",
    "RandomShift",
    "RandomRotate",
    "SortByMotion",
    "DenoiseByLength",
    "DenoiseBySpread",
    "DenoiseByMotion",
    "MergeBodies",
    "RemoveZeroFrames",
    "Denoise",
]


class Compose:
    """Composes several transforms together. This transform does not support torchscript.
    Please, see the note below.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.PILToTensor(),
        >>>     transforms.ConvertImageDtype(torch.float),
        >>> ])

    .. note::
        In order to script the transformations, please use ``torch.nn.Sequential`` as below.

        >>> transforms = torch.nn.Sequential(
        >>>     transforms.CenterCrop(10),
        >>>     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>> )
        >>> scripted_transforms = torch.jit.script(transforms)

        Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor``, does not require
        `lambda` functions or ``PIL.Image``.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class PadBodies:
    """Pad each frame with zeros to match number of bodies.

    Args:
        max_bodies (int): number of bodies to pad to
    """

    def __init__(self, max_bodies, **kwargs):
        self.max_bodies = max_bodies
        self.pad_kwargs = kwargs

    def __call__(self, x):
        return F.pad_bodies(x, max_bodies=self.max_bodies, **self.pad_kwargs)


class PadFrames:
    """Pad sequence with zeros to match number of frames.

    Args:
        max_frames (int): number of frames to pad to
        axis (int): time dimension axis, default is 1
    """

    def __init__(self, max_frames, axis=1, **kwargs):
        self.max_frames = max_frames
        self.pad_kwargs = kwargs
        self.axis = axis

    def __call__(self, x):
        return F.pad_frames(
            x, max_frames=self.max_frames, axis=self.axis, **self.pad_kwargs
        )


class SampleFrames:
    """Uniformly sample frames from sequence.

    Args:
        num_frames (int): number of frames to sample
        num_clips (int): number of clips to sample
        axis (int): axis to stack sampled clips
    """

    def __init__(self, num_frames, num_clips=1, axis=0) -> None:
        self.num_frames = num_frames
        self.num_clips = num_clips
        self.axis = axis

    def __call__(self, x):
        return F.sample_frames(
            x, num_frames=self.num_frames, num_clips=self.num_clips, axis=self.axis
        )


class SelectKBodies:
    """Select K bodies from sequeqnce based on motion.

    Args:
        k (int): number of bodies to sort
    """

    def __init__(self, k):
        self.k = k

    def __call__(self, x):
        return F.select_k_bodies(x, k=self.k)


class SplitFrames:
    """Split frame into N frames each containing one skeleton."""

    def __call__(self, x):
        return F.split_frames(x)


class CenterJoint:
    """Center skeleton on joint.

    Args:
        joint_id (int): joint id to center skeleton
        all (bool): set to False to center along initial frame
    """

    def __init__(self, joint_id=1, all=False):
        self.joint_id = joint_id
        self.all = all

    def __call__(self, x):
        return F.sub_center_joint(x, joint_id=self.joint_id, all=self.all)


class ParallelBone:
    """Transform skeleton so two joints are parallel to a certain axis.

    Args:
        first_id (int): first joint id
        second_id (int): second joint id
        axis (int): axis to be parallel to 0, 1, 2, corresponds to x, y, z
    """

    def __init__(self, first_id, second_id, axis=2) -> None:
        super().__init__()

        self.first_id = first_id
        self.second_id = second_id
        self.axis = axis

    def __call__(self, x):
        return F.parallel_bone(
            x, first_id=self.first_id, second_id=self.second_id, axis=self.axis
        )


class RandomShift:
    """Randomly shift skeleton sequences.

    Args:
        low (int): minimum shift
        high (int): maximum shift
    """

    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, x):
        return F.random_shift(x, low=self.low, high=self.high)


class RandomRotate:
    """Randomly rotate skeleton sequences.

    Args:
        degrees (int): range of shift in degrees

    Example:
        # randomly select angles from -30 ~ 30 degrees and rotate
        >>> transforms.RandomRotate(30)
    """

    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, x):
        return F.random_rotate(x, self.degrees)


class SortByMotion:
    """Sort skeletons in a single frame by motion."""

    def __call__(self, x):
        return F.select_k_bodies(x, k=x.shape[0])


class DenoiseByLength:
    """Denoise bodies by length of non zero frames.

    Removes bodies whose length is under a minimum length
    """

    def __call__(self, x):
        return F.denoising_by_length(x)


class DenoiseBySpread:
    """Denoise bodies by length of frames under a certain x, y spread.

    Removes bodies where noisy frames exceed a certain ratio.
    Frames with higher x, y spread than the threshold are considered noisy.
    """

    def __call__(self, x):
        return F.denoising_by_spread(x)


class DenoiseByMotion:
    """Denoise bodies by filtering frames within a certain range of motion.

    Valid frames by spread are selected to compute the motion of a body.
    Bodies whose motion is outside a certain low, high range is removed.
    """

    def __call__(self, x):
        return F.denoising_by_motion(x)


class MergeBodies:
    """Merges different bodies who don't overlap into 2 actors.

    Bodies are expected to be sorted by motion.
    First body is selected as the main actor as it has the highest motion.
    subsequent bodies are compared with the main actor.
    If there aren't overlapping frames, the body is merged with the main actor.
    If there are overlapping frames, try to merge with the second actor.
    If there are overlapping frames with the second actor, the body is removed.
    """

    def __call__(self, x):
        if x.shape[0] > 1:
            x = F.merge_bodies(x)
            x = F.remove_zero_frames(x)

        return x


class RemoveZeroFrames:
    """Remove frames where all bodies are zero from the sequence."""

    def __call__(self, x):
        return F.remove_zero_frames(x)


class Denoise:
    """Denoise skeleton sequence used on NTU."""

    def __call__(self, x):
        if x.shape[0] > 1:
            x = F.select_k_bodies(x, k=x.shape[0])

            x = F.denoising_by_length(x)

            if x.shape[0] > 1:
                x = F.denoising_by_spread(x)

                if x.shape[0] > 1:
                    x = F.denoising_by_motion(x)

            if x.shape[0] > 1:
                x = F.merge_bodies(x)

            x = F.remove_zero_frames(x)

        return x
