"""
Author: PointNeXt
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from easydict import EasyDict as edict
from openpoints.utils import registry
from openpoints.transforms import build_transforms_from_cfg

DATASETS = registry.Registry('dataset')

class CustomDataset(Dataset):
    def __init__(self, data_path, file_list_path, class_names_path, transform=None):
        self.data_path = data_path
        self.file_list_path = file_list_path
        self.transform = transform
        self.files = self.load_files()
        self.classes = self.load_class_names(class_names_path)

    def load_files(self):
        files = []
        with open(self.file_list_path, 'r') as f:
            lines = [line.strip() for line in f]
            for line in lines:
                class_name, file_name = os.path.split(line)
                files.append((class_name, file_name))
        return files

    def load_class_names(self, class_names_path):
        with open(class_names_path, 'r') as f:
            class_names = [line.strip() for line in f]
        return class_names

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        class_name, file_name = self.files[idx]
        file_path = os.path.join(self.data_path, class_name, f"{file_name}.txt")
        data = self.load_data(file_path)
        data_dict = {'pos': data, 'x': data}
        if self.transform:
            data_dict = self.transform(data_dict)
        return data_dict

    def load_data(self, file_path):
        data = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                values = line.strip().split(',')
                data.append([float(v) for v in values])
        return np.array(data, dtype=np.float32)

def concat_collate_fn(datas):
    """collate fn for point transformer
    """
    pts, feats, labels, offset, count, batches = [], [], [], [], 0, []
    for i, data in enumerate(datas):
        count += len(data['pos'])
        offset.append(count)
        pts.append(data['pos'])
        feats.append(data['x'])
        labels.append(data['y'])
        batches += [i] *len(data['pos'])
        
    data = {'pos': torch.cat(pts), 'x': torch.cat(feats), 'y': torch.cat(labels),
            'o': torch.IntTensor(offset), 'batch': torch.LongTensor(batches)}
    return data


def build_dataset_from_cfg(cfg, default_args=None):
    """
    Build a dataset, defined by dataset_name.
    Args:
        cfg (eDICT):
    Returns:
        Dataset: a constructed dataset specified by dataset_name.
    """
    return DATASETS.build(cfg, default_args=default_args)

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def build_dataloader_from_cfg(batch_size,
                              dataset_cfg=None,
                              dataloader_cfg=None,
                              datatransforms_cfg=None,
                              split='train',
                              distributed=True,
                              dataset=None
                              ):
    if dataset is None:
        if datatransforms_cfg is not None:
            # in case only val or test transforms are provided. 
            if split not in datatransforms_cfg.keys() and split in ['val', 'test']:
                trans_split = 'val'
            else:
                trans_split = split
            data_transform = build_transforms_from_cfg(trans_split, datatransforms_cfg)
        else:
            data_transform = None

        if split not in dataset_cfg.keys() and split in ['val', 'test']:
            dataset_split = 'test' if split == 'val' else 'val'
        else:
            dataset_split = split
        split_cfg = dataset_cfg.get(dataset_split, edict())
        if split_cfg.get('split', None) is None:    # add 'split' in dataset_split_cfg
            split_cfg.split = split
        split_cfg.transform = data_transform
        dataset = build_dataset_from_cfg(dataset_cfg.common, split_cfg)
        """
        file_list_path = os.path.join(dataset_cfg.common.data_dir, 'list_test_modelnet40.txt' if split == 'test' else 'filelist.txt')
        class_names_path = os.path.join(dataset_cfg.common.data_dir, 'modelnet40_shape_names.txt')

        dataset = CustomDataset(data_path=dataset_cfg.common.data_dir,
                                file_list_path=file_list_path,
                                class_names_path=class_names_path,
                                transform=data_transform)
        """

    collate_fn = dataset.collate_fn if hasattr(dataset, 'collate_fn') else None
    collate_fn = dataloader_cfg.collate_fn if dataloader_cfg.get('collate_fn', None) is not None else collate_fn
    collate_fn = eval(collate_fn) if isinstance(collate_fn, str) else collate_fn

    shuffle = split == 'train'
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 num_workers=int(dataloader_cfg.num_workers),
                                                 worker_init_fn=worker_init_fn,
                                                 drop_last=split == 'train',
                                                 sampler=sampler,
                                                 collate_fn=collate_fn, 
                                                 pin_memory=True
                                                 )
    else:
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 num_workers=int(dataloader_cfg.num_workers),
                                                 worker_init_fn=worker_init_fn,
                                                 drop_last=split == 'train',
                                                 shuffle=shuffle,
                                                 collate_fn=collate_fn,
                                                 pin_memory=True)
    return dataloader