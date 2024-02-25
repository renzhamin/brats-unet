import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
import os

def get_data_loader(data_dir, transforms=None, batch_size=8, num_workers=2):
    dataset = BratsDataset(data_dir=data_dir, transforms=transforms)
    return DataLoader(dataset, batch_size=batch_size,
                      num_workers=num_workers, shuffle=True)

class BratsDataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        self.data_dir = data_dir
        self.transforms = transforms
        self.cases = os.listdir(data_dir)

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case_path = os.path.join(self.data_dir, self.cases[idx])
        modalities = ['flair', 't1', 't1ce', 't2']
        images = []
        for modality in modalities:
            img_path = os.path.join(case_path, f'{self.cases[idx]}_{modality}.nii.gz')
            img = nib.load(img_path).get_fdata().T
            images.append(img)
        images = np.stack(images, axis=0)
        
        seg_path = os.path.join(case_path, f'{self.cases[idx]}_seg.nii.gz')
        seg = nib.load(seg_path).get_fdata().T

        images, seg = torch.tensor(images, dtype=torch.long), torch.tensor(seg, dtype=torch.long)
        
        if self.transforms:
            images = self.transforms(images)
            seg = self.transforms(seg)

        return images, seg
