from base import BaseDataLoader
from dataset.datasets import PatchedDataset
from torchvision import transforms
from torch.utils.data.sampler import SequentialSampler


class PatchedDataLoader(BaseDataLoader):
    def __init__(
        self,
        data_dir,
        patch_size,
        batch_size,
        patch_stride=None,
        preds=None,
        target_dist=0.0,
        shuffle=True,
        validation_split=0.0,
        num_workers=1,
        training=True
    ):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.3551, 0.4698, 0.2261),
                                 (0.1966, 0.1988, 0.1761))
        ])
        target_trsfm = transforms.Compose([
            transforms.ToTensor(),
        ])
        rand_trsfm = transforms.RandomApply([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip()
        ])
        self.data_dir = data_dir
        self.dataset = PatchedDataset(
            self.data_dir,
            patch_size,
            patch_stride=patch_stride,
            preds=preds,
            target_dist=target_dist,
            transform=trsfm,
            target_transform=target_trsfm,
            rand_transform=rand_trsfm if training and shuffle else None,
            train=training,
            late_init=True
        )
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    def _split_sampler(self, split):
        train_sampler, valid_sampler = super()._split_sampler(split)

        if valid_sampler is not None:
            self.dataset.make_dataset(valid_indices=valid_sampler.indices)
        else:
            self.dataset.make_dataset()

        train_idx, valid_idx = [], []
        for patch in self.dataset.patches:
            if valid_sampler is not None and patch.idx in valid_sampler.indices:
                valid_idx.append(self.dataset.patches.index(patch))
            else:
                train_idx.append(self.dataset.patches.index(patch))

        if valid_sampler is not None:
            train_sampler.indices, valid_sampler.indices = train_idx, valid_idx
        else:
            train_sampler = SequentialSampler(train_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def update_dataset(self, preds):
        self.dataset.preds = preds
        self.dataset.patches.clear()
        self.n_samples = len(self.dataset)

        train_sampler, valid_sampler = self._split_sampler(
            self.validation_split)
        if valid_sampler is not None:
            self.valid_sampler.indices = valid_sampler.indices
        self.sampler.indices = train_sampler.indices
