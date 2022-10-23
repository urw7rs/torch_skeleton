from . import functions


class Denoise:
    def __call__(self, x):
        if x.shape[0] > 1:
            x = functions.select_k_bodies(x, k=x.shape[0])

            x = functions.denoising_by_length(x)

            if x.shape[0] > 1:
                x = functions.denoising_by_spread(x)

                if x.shape[0] > 1:
                    x = functions.denoising_by_motion(x)

            if x.shape[0] > 1:
                x = functions.merge_bodies(x)

            x = functions.remove_zero_frames(x)

        return x


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
        if x.shape[0] > 1:
            x = functions.merge_bodies(x)
            x = functions.remove_zero_frames(x)

        return x


class RemoveZeroFrames:
    def __call__(self, x):
        return functions.remove_zero_frames(x)
