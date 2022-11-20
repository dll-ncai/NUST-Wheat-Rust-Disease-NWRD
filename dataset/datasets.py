from typing import Any, Callable, Optional
import torch.nn.functional as F
from base import BaseDataset
from utils.util import TransformMultiple, pil_loader
from dataset.patches import Patches


class PatchedDataset(BaseDataset):
    def __init__(
        self,
        root: str,
        patch_size: int,
        patch_stride: int = None,
        preds: list = None,
        target_dist: int = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        rand_transform: Optional[Callable] = None,
        train: bool = True,
        late_init: bool = False
    ) -> None:
        super().__init__(root, pil_loader, transforms, transform, target_transform, train)
        self.patches = Patches(patch_size, patch_stride)
        self.preds = preds
        self.target_dist = target_dist / patch_size ** 2 if target_dist is not None else 0
        self.rand_transform = TransformMultiple(rand_transform)
        if not late_init:
            self.make_dataset()

    def make_dataset(self, valid_indices=[]):
        for idx in range(super().__len__()):
            _, mask = super().__getitem__(idx)
            if self.preds is not None:
                mask = self._union_mask(mask, self.preds[idx])
            self.patches.create(idx, mask, cond_fn=self._dist_fn,
                                overlap=idx not in valid_indices)

    def __getitem__(self, index: int) -> Any:
        patch = self.patches[index]
        img, mask = super().__getitem__(patch.idx)

        img_patch = self.patches.get_patch(img, patch)
        mask_patch = self.patches.get_patch(mask, patch)
        img_patch, mask_patch = self.rand_transform(
            (img_patch, mask_patch.unsqueeze(dim=0)))
        return img_patch, mask_patch.squeeze(dim=0)

    def _union_mask(self, mask, pred):
        pred = F.pad(
            pred, (0, mask.shape[1] - pred.shape[1], 0, mask.shape[0] - pred.shape[0]))
        return (mask + pred) - (mask * pred)

    def _dist_fn(self, mask, patch):
        data = self.patches.get_patch(mask, patch)
        return data.count_nonzero() > self.target_dist
