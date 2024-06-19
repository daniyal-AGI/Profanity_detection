from torch.utils.data import Dataset
from utils.data_transformation import time_shift, spectro_augment
import pandas as pd
import torch
import os
import numpy as np


class AudioDataset (Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd. read_csv (csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self) :
        return len(self.annotations)

    def __getitem__(self, index) :

        file_path = os.path.join(self. root_dir, self.annotations. iloc[index, 0])
        #print(file_path)
        audio_feature= np.load(file_path)
        
        y_label = self .annotations. iloc [index, 1]
        
        if self.transform:
            audio_feature = self.transform(audio_feature)
        
        audio_feature=torch.from_numpy(audio_feature)

        audio_feature, sr = time_shift((audio_feature, 16000), 0.1)
        audio_feature = audio_feature.unsqueeze(0)
        audio_feature = spectro_augment(audio_feature, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
        audio_feature = audio_feature.squeeze()

        max_pool, _ = torch.max(audio_feature, dim=1)

        return (max_pool, y_label)