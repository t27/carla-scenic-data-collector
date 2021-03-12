import sys

sys.path.insert(0, "..")
from generate_segment_trajectories import get_dtw_maps

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

normal_rounds = ["round_0"]  # , "round_1", "round_2", "round_3", "round_4"]
anomalous_rounds = [
    # ["debris_round2"]
    # "tl_sl2_round_0",
    # "tl_sl2_round_1",
    # "tl_sl2_round_2",
    "tl_sl_round_0",
    "tl_sl_round_1",
    "tl_sl_round_2",
    "tl_sl_round_3",
]
anomalous_len = 1000  # 426  # number of scenarios for a 25 frame segment
max_real_len = 1000
num_channels = 10
means = [0.5 for i in range(num_channels)]
stds = [0.5 for i in range(num_channels)]
transform = transforms.Normalize(mean=means, std=stds)


class DtwDataset(Dataset):
    def __init__(self, data, transforms=None) -> None:
        super().__init__()
        self.data = data
        if transforms is None:
            self.transforms = transform
        # breakpoint()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        return self.transforms(torch.tensor(self.data[index]))


def get_anomalous_train_test_loaders(batch_size=32, train_ratio=0.7):
    anomalous_data_count = anomalous_len  # since normal count is greater that anomalous, calculate the splits based on the anomalous data
    train_count = int(train_ratio * anomalous_data_count)
    test_count = anomalous_data_count - train_count
    anomalous_data = get_dtw_maps(
        rounds=anomalous_rounds, max_dtw_maps=anomalous_data_count
    )
    train_dataset = DtwDataset(anomalous_data[:train_count])
    dev_dataset = DtwDataset(anomalous_data[train_count:])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )

    return train_loader, dev_loader


def get_real_train_test_loaders(batch_size=32, train_ratio=0.7, max_len=None):
    if max_len is None:
        real_data_count = max_real_len  # since normal count is greater that real, calculate the splits based on the real data
    else:
        real_data_count = max_len
    train_count = int(train_ratio * real_data_count)
    test_count = real_data_count - train_count
    real_data = get_dtw_maps(rounds=normal_rounds, max_dtw_maps=real_data_count)
    train_dataset = DtwDataset(real_data[:train_count])
    dev_dataset = DtwDataset(real_data[train_count:])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )

    return train_loader, dev_loader


# len(anomalous)
# 426
def get_train_test_loaders(batch_size=32, train_ratio=0.7, only_real=False):

    if not only_real:
        anomalous_data_count = anomalous_len  # since normal count is greater that anomalous, calculate the splits based on the anomalous data
        train_count = int(train_ratio * anomalous_data_count)
    else:
        anomalous_data_count = max_real_len  # since normal count is greater that anomalous, calculate the splits based on the anomalous data
        train_count = int(train_ratio * anomalous_data_count)

    test_count = anomalous_data_count - train_count
    real_data = get_dtw_maps(rounds=normal_rounds, max_dtw_maps=anomalous_data_count)
    if not only_real:
        anomalous_data = get_dtw_maps(
            rounds=anomalous_rounds, max_dtw_maps=anomalous_data_count
        )
        train_dataset = DtwDataset(
            real_data[:train_count] + anomalous_data[:train_count]
        )
        dev_dataset = DtwDataset(real_data[train_count:] + anomalous_data[train_count:])
    else:
        train_dataset = DtwDataset(real_data[:train_count])
        dev_dataset = DtwDataset(real_data[train_count:])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )

    return train_loader, dev_loader


if __name__ == "__main__":
    ds = DtwDataset()
