import glob
from pathlib import Path
from typing import Any, Callable, Optional
from torchvision.datasets import VisionDataset


class BaseDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        loader: Callable[[str], Any],
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        train: bool = True
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.root_path = Path(root)
        self.loader = loader

        mode = 'train' if train else 'test'
        self.data = sorted(glob.glob(f'{mode}/images/*.jpg', root_dir=root))
        self.masks = sorted(glob.glob(f'{mode}/masks/*.png', root_dir=root))

    def __getitem__(self, index: int) -> Any:
        img_path, mask_path = self.data[index], self.masks[index]
        img_path, mask_path = self.root_path / img_path, self.root_path / mask_path

        img, mask = self.loader(img_path), self.loader(mask_path)
        img, mask = self.transforms(img, mask)
        return img, mask.squeeze(dim=0).bool().float()

    def __len__(self) -> int:
        return len(self.data)
