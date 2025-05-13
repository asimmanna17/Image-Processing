import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import random
import torch


class DatasetNpy_Mix(Dataset):
    def __init__(self, data_folder, patch_size, training=True, num_patches_per_image=4):
        self.data_folder = data_folder
        self.img_list = os.listdir(data_folder)
        self.patch_size = patch_size
        self.training = training
        self.num_patches_per_image = num_patches_per_image

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        npypath = os.path.join(self.data_folder, self.img_list[index])
        imdata = np.load(npypath)
        sht = imdata['sht']  # shape [8,H,W]
        mid = imdata['mid']
        lng = imdata['lng']
        hdr = imdata['hdr']  # shape [4,H,W]

        if self.training:
            sht_patches, mid_patches, lng_patches, hdr_patches = [], [], [], []

            for _ in range(self.num_patches_per_image):
                # Concatenate all channels for cropping
                imstack = np.concatenate([sht, mid, lng, hdr], axis=0)
                crop_stack = self.random_crop(imstack)

                sht_patches.append(self.to_tensor(crop_stack[0:8]))
                mid_patches.append(self.to_tensor(crop_stack[8:16]))
                lng_patches.append(self.to_tensor(crop_stack[16:24]))
                hdr_patches.append(self.to_tensor(crop_stack[24:]))

            # Stack patches into shape (num_patches, channels, H, W)
            return {
                'sht': torch.stack(sht_patches),
                'mid': torch.stack(mid_patches),
                'lng': torch.stack(lng_patches),
                'hdr': torch.stack(hdr_patches)
            }

        else:
            # Just convert all to tensor for eval
            return {
                'sht': self.to_tensor(sht),
                'mid': self.to_tensor(mid),
                'lng': self.to_tensor(lng),
                'hdr': self.to_tensor(hdr),
                'save_name': os.path.splitext(self.img_list[index])[0]
            }

    def to_tensor(self, np_array):
        return torch.from_numpy(np_array).float()

    def random_crop(self, np_array):
        c, h, w = np_array.shape
        assert c == 28
        w_start = random.randint(0, w - self.patch_size)
        h_start = random.randint(0, h - self.patch_size)
        return np_array[:, h_start:h_start + self.patch_size, w_start:w_start + self.patch_size]


if __name__ == '__main__':
    folderpath = '/data/asim/ISP/HDR_transformer/data/RAW'
    dataset = DatasetNpy_Mix(data_folder=folderpath, patch_size=128, training=True, num_patches_per_image=4)
    train_loader = DataLoader(dataset=dataset, batch_size=2, shuffle=True, num_workers=8)

    for i, cur_data in enumerate(train_loader):
        sht = cur_data['sht']  # shape [1, num_patches, 8, 128, 128]
        mid = cur_data['mid']
        lng = cur_data['lng']
        hdr = cur_data['hdr']

        print(f"{i}: sht {sht.shape}, mid {mid.shape}, lng {lng.shape}, hdr {hdr.shape}")
        
