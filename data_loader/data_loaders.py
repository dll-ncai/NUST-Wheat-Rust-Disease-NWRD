from base import BaseDataLoader
from dataset.datasets import PatchedDataset
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


class PatchedDataLoader(BaseDataLoader):
    def __init__(
        self,
        data_dir,
        patch_size,
        batch_size,
        shuffle=True,
        validation_split=0.0,
        num_workers=1,
        training=True
    ):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.3551, 0.4698, 0.2261), (0.1966, 0.1988, 0.1761))
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
            transform=trsfm,
            target_transform=target_trsfm,
            rand_transform=rand_trsfm if training and shuffle else None,
            train=training
        )
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    def _split_sampler(self, split):
        train_sampler, valid_sampler = super()._split_sampler(split)

        train_idx, valid_idx = [], []
        for patch in self.dataset.patches:
            if valid_sampler is not None and patch.idx in valid_sampler.indices:
                valid_idx.append(self.dataset.patches.index(patch))
            else:
                train_idx.append(self.dataset.patches.index(patch))

        if valid_sampler is not None:
            train_sampler.indices, valid_sampler.indices = train_idx, valid_idx
        else:
            train_sampler = SubsetRandomSampler(train_idx)

        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler
