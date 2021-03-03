import cv2
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset


class DetectionDataset(Dataset):

    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()

        self.image_ids = dataframe["image_id"].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        records = self.df[(self.df['image_id'] == image_id)]
        records = records.reset_index(drop=True)

        file = f"{self.image_dir}/{image_id}.png"
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

        image = np.stack([image, image, image])
        image = image.astype('float32')
        image = image.transpose(1, 2, 0)

        if records.loc[0, "class_id"] == 0:
            records = records.loc[[0], :]

        boxes = records[['x_min', 'y_min', 'x_max', 'y_max']].values
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)
        labels = torch.tensor(records["class_id"].values, dtype=torch.int64)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']

            target['boxes'] = torch.tensor(sample['bboxes'])

        if target["boxes"].shape[0] == 0:
            # Albumentation cuts the target (class 14, 1x1px in the corner)
            target["boxes"] = torch.from_numpy(
                np.array([[0.0, 0.0, 1.0, 1.0]]))
            target["area"] = torch.tensor([1.0], dtype=torch.float32)
            target["labels"] = torch.tensor([0], dtype=torch.int64)

        return image, target

    def __len__(self):
        return self.image_ids.shape[0]


def read_data(path):
    df = pd.read_csv(path)
    df.fillna(0, inplace=True)
    df.loc[df["class_id"] == 14, ['x_max', 'y_max']] = 1.0

    df["class_id"] = df["class_id"] + 1
    df.loc[df["class_id"] == 15, ["class_id"]] = 0
    return df
