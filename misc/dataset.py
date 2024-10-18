import torch
import pandas as pd
import numpy as np
import albumentations as A
import os

from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from utility.dataset import dilate_pixel


class CustomDataset(Dataset):
    def __init__(self, args, df, data_type, transform=None):
        super().__init__()
        self.args = args
        self.df = df.reset_index()
        self.data_type = data_type
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        self.df = self.df.fillna(0)
        image_name = self.df['image_name'][idx]
        segments_dir = f'{self.args.image_segment_path}/{self.data_type}/{image_name.split(".")[0]}'
        
        segments = []
        for i in range(5):
            segment_path = os.path.join(segments_dir, f'segment_{i+1}.png')
            segment = np.array(Image.open(segment_path).convert("RGB"))
            segments.append(segment)
        
        label_list = [0] * (self.args.output_channel * 2)
        masks = [np.zeros((self.args.output_channel, self.args.image_resize, self.args.image_resize)) for _ in range(5)]

        for i in range(self.args.output_channel):
            y = int(round(self.df[f'label_{i}_y'][idx]))
            x = int(round(self.df[f'label_{i}_x'][idx]))
            if y != 0 and x != 0:
                label_list[2 * i] = y
                label_list[2 * i + 1] = x
                box_index = self.get_box_index(i)  # Determine which segment the landmark belongs to
                # print(image_name)
                masks[box_index][i] = dilate_pixel(self.args, masks[box_index][i], y, x)
            
        if self.transform:
            augmented_segments = []
            augmented_masks = []
            for segment, mask in zip(segments, masks):
                augmentations = self.transform(image=segment, masks=[mask[ch] for ch in range(self.args.output_channel)])
                augmented_segments.append(augmentations["image"])
                augmented_masks.append(torch.stack(augmentations["masks"], dim=0))
            segments = augmented_segments
            masks = augmented_masks

        return torch.stack(segments, dim=0), torch.stack(masks, dim=0), segments_dir, image_name, label_list

    def get_box_index(self, landmark_idx):
        if landmark_idx in [0, 1]:
            return 0
        elif landmark_idx == 2:
            return 1
        elif landmark_idx in [3, 4, 5, 6, 7, 8]:
            return 2
        elif landmark_idx == 9:
            return 3
        elif landmark_idx == 10:
            return 4
        else:
            return -1

def load_data(args):
    print("---------- Starting Loading Dataset ----------")
    IMAGE_RESIZE = args.image_resize
    BATCH_SIZE = args.batch_size

    # Load the separate CSV files for training and validation datasets
    train_df = pd.read_csv(args.train_csv_postprocessed)
    val_df = pd.read_csv(args.val_csv_postprocessed)
    test_df  = pd.read_csv(args.test_csv_postprocessed)

    # if args.augmentation: 
    #     transform = A.Compose([
    #         A.Resize(height=IMAGE_RESIZE, width=IMAGE_RESIZE),
    #         A.Rotate(limit=20, p=0.5),
    #         A.Normalize(
    #             mean=(0.485, 0.456, 0.406),
    #             std=(0.229, 0.224, 0.225),
    #         ),
    #         ToTensorV2(),
    #     ], is_check_shapes=False)
    # else:
    transform = A.Compose([
        A.Resize(height=IMAGE_RESIZE, width=IMAGE_RESIZE),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ], is_check_shapes=False)

    train_dataset = CustomDataset(
        args, train_df, 'train', transform
    )
    val_dataset = CustomDataset(
        args, val_df, 'validation', transform
    )
    test_dataset = CustomDataset(
        args, test_df, 'test', transform
    )
    print('len of train dataset: ', len(train_dataset))
    print('len of val dataset: ', len(val_dataset))
    print('len of test dataset: ', len(test_dataset))

    num_workers = 4 * torch.cuda.device_count()
    train_loader = DataLoader(
        train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, batch_size=1, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, shuffle=False, batch_size=1, num_workers=num_workers
    )

    print("---------- Loading Dataset Done ----------")

    return train_loader, val_loader, test_loader