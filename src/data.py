"""CIFAR-10: загрузка, аугментация, разбиение train/val/test."""
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

CLASSES = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
MEAN = STD = (0.5, 0.5, 0.5)   # как при обучении исходных весов: пиксели -> [-1, 1]


def _eval_tf():
    return T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])


def _train_tf():
    return T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4),
                      T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])


def get_loaders(root="data", val_split=0.2, batch_size=64, num_workers=2, seed=42, download=True):
    """Возвращает train/val/test. Аугментация только на train: val размечается из отдельного
    экземпляра датасета без аугментации (в исходной версии аугментация случайно попадала и в val)."""
    train_full = torchvision.datasets.CIFAR10(root, train=True, download=download, transform=_train_tf())
    val_full = torchvision.datasets.CIFAR10(root, train=True, download=download, transform=_eval_tf())
    test = torchvision.datasets.CIFAR10(root, train=False, download=download, transform=_eval_tf())

    n_val = int(len(train_full) * val_split)
    perm = torch.randperm(len(train_full), generator=torch.Generator().manual_seed(seed)).tolist()
    train_ds, val_ds = Subset(train_full, perm[n_val:]), Subset(val_full, perm[:n_val])

    mk = lambda ds, shuffle: DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)  # noqa: E731
    return mk(train_ds, True), mk(val_ds, False), mk(test, False)


class _CIFAR10Test(torchvision.datasets.CIFAR10):
    """CIFAR-10, для которого проверяется только тестовый файл: для оценки не нужно качать 5 train-батчей."""

    def _check_integrity(self) -> bool:
        return all(torchvision.datasets.utils.check_integrity(
            str(__import__("pathlib").Path(self.root) / self.base_folder / f), m) for f, m in self.test_list + [("batches.meta", self.meta["md5"])])


def get_test_loader(root="data", batch_size=64, num_workers=2, download=True):
    test = _CIFAR10Test(root, train=False, download=download, transform=_eval_tf())
    return DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
