import numpy as np


class Vector2d:
    def __init__(self, v: np.ndarray, keep_dim=False, from_3d=False):  # ( n, -1 )
        self.keep_dim = keep_dim

        if from_3d:
            self.v = v.reshape(v.shape[0], -1, 3).transpose(0, 2, 1)[:, :2]
        else:
            self.v = v.reshape(v.shape[0], -1, 2).transpose(0, 2, 1)

        if self.v.shape[-1] == 1:
            self.v = self.v.squeeze(axis=-1)

    def __call__(self):
        return self.v

    @property
    def x(self):
        return self.v[:, [0]] if self.keep_dim else self.v[:, 0]

    @property
    def y(self):
        return self.v[:, [1]] if self.keep_dim else self.v[:, 1]

    @property
    def angle(self):
        assert len(self.v.shape) == 2, "Not a vector"

        return -np.rad2deg(np.arctan2(self.y, self.x)) + 180  # ( n, 1 )

    @property
    def norm(self):
        assert len(self.v.shape) == 2, "Not a vector"

        n = np.linalg.norm(self.v, ord=2, axis=-1)  # ( n, )

        return n[:, np.newaxis] if self.keep_dim else n
