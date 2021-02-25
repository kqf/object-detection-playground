import albumentations as alb
from albumentations.pytorch import ToTensorV2


_mean = [0, 0, 0]
_std = [1, 1, 1]

bbox_params = {
    'format': 'pascal_voc',
    'label_fields': ['labels']
}


def transform(train=True, mean=None, std=None):
    normalize = alb.Compose([
        alb.Normalize(mean=_mean, std=_std,
                      max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0)
    ], bbox_params=bbox_params)

    if not train:
        return normalize

    return alb.Compose([
        alb.Flip(0.5),
        alb.ShiftScaleRotate(scale_limit=0.1, rotate_limit=45, p=0.25),
        alb.LongestMaxSize(max_size=800, p=1.0),
        normalize,

    ], bbox_params=bbox_params)