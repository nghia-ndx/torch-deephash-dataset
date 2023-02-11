# Torch DeepHash Dataset

A simple PyTorch implementation of common datasets for Deep Hashing. For now, only NUS-WIDE and COCO datasets are supported. 

# Installation
Install via `pip`
```
pip install git+https://github.com/nghia-ndx/torch-deephash-dataset
```

# Usages
- Each dataset supports 3 types of splits: 
    - `train`: Train set
    - `test`: Test set
    - `db`: Full dataset (exclude train and test)
- On first `__init__`, the dataset will be downloaded to the specified `root` location and will be reused on future runs. 
```python
from torch_deephash_dataset.coco import COCODataset
from torch_deephash_dataset.nus_wide import NUSWIDEDataset

coco_dataset = COCODataset('datasets/coco', split='train')
nus_wide_dataset = NUSWIDEDataset('datasets/nus_wide', split='test')
```
- Subsequently, if you want to re-download, use `force_download=True`. 

    **Note**: using `force_download=True` will DELETE everything in the specified root directory before re-downloading.

```python
from torch_deephash_dataset.nus_wide import NUSWIDEDataset

nus_wide_dataset = NUSWIDEDataset(
    'datasets/nus_wide', 
    split='test', 
    force_download=True
)
```

- The dataset could be used with PyTorch's `DataLoader` as usual:
```python
from torch.utils.data import DataLoader

data_loader = DataLoader(
    dataset=nus_wide_dataset, 
    batch_size=64, 
    num_workers=4
)
```

- Data transform and label transform can be appied via `transform` and `target_tranform` arguments:
```python
from torch_deephash_dataset.nus_wide import NUSWIDEDataset
from torchvision import transforms

nus_wide_dataset = NUSWIDEDataset(
    'datasets/nus_wide', 
    split='test', 
    transform=transforms.ToTensor()
)
```