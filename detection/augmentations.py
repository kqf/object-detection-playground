import cv2
import albumentations as alb
from albumentations.pytorch import ToTensorV2


_mean = [0, 0, 0]
_std = [1, 1, 1]

bbox_params = {
    'format': 'yolo',
    'label_fields': ['labels'],
    'min_visibility': 0.4,
}


def transform(train=True, mean=None, std=None, scale=1., size=2000):
    normalize = alb.Compose([
        alb.PadIfNeeded(
            min_height=int(size * scale),
            min_width=int(size * scale),
            border_mode=cv2.BORDER_CONSTANT,
        ),
        alb.Normalize(mean=_mean, std=_std,
                      max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0)
    ], bbox_params=bbox_params)

    if not train:
        return normalize

    return alb.Compose([
        normalize,

    ], bbox_params=bbox_params)
