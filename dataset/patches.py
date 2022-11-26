from collections import UserList
from torchvision.utils import make_grid


class Patch():
    def __init__(self, idx, x, y) -> None:
        self.idx = idx
        self.x, self.y = x, y
        self.data = None

    def __eq__(self, __o: object) -> bool:
        return self.idx == __o.idx and self.x == __o.x and self.y == __o.y


class Patches(UserList):
    def __init__(self, size, stride=None):
        super().__init__()
        self.size = size
        self.stride = stride if stride is not None else size

    def create(self, index, data, cond_fn=None, no_overlap=False):
        stride = self.size if no_overlap else self.stride
        for x in range(0, data.size(-2) - self.size + 1, stride):
            for y in range(0, data.size(-1) - self.size + 1, stride):
                patch = Patch(index, x, y)
                if cond_fn is None or cond_fn(data, patch):
                    self.append(patch)

    def get_patch(self, data, patch: Patch):
        assert data.ndim in {2, 3}, 'only 2-D and 3-D Tensors are supported.'
        _data = data.unsqueeze(dim=0) if data.ndim == 2 else data
        data_patch = _data[:, patch.x:patch.x + self.size,
                           patch.y:patch.y + self.size]
        return data_patch.squeeze(dim=0) if data.ndim == 2 else data_patch

    def store_data(self, indices, data):
        for idx in range(len(indices)):
            self[indices[idx]].data = [data[i][idx] for i in range(len(data))]

    def retrieve_data(self, indices):
        return [[self[idx].data[i] for idx in indices] for i in range(len(self[indices[0]].data))]

    def fuse_data(self, index: int, data_idx: int):
        indices = [self.index(patch) for patch in self if patch.idx == index]
        indices.sort(key=lambda idx: (self[idx].x, self[idx].y))

        data = self.retrieve_data(indices)
        nrow = sum([self[idx].x == 0 for idx in indices])
        return make_grid(data[data_idx], nrow, padding=0)
