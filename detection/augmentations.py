# import cv2
import albumentations as alb
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import DualTransform


_mean = [0, 0, 0]
_std = [1, 1, 1]

bbox_params = {
    'format': 'yolo',
    'label_fields': ['labels'],
    'min_visibility': 0.4,
}


class DebugAugmentations(DualTransform):
    def __init__(self, always_apply=True, p=1):
        super().__init__(always_apply, p)

    def get_params(self):
        return {"scale": 1}

    def apply(self, img, *args, **params):
        return img

    def apply_to_bbox(self, bbox, **params):
        import ipdb; ipdb.set_trace(); import IPython; IPython.embed() # noqa
        return bbox

    def apply_to_keypoint(self, keypoint, scale=0, **params):
        return keypoint

    def get_transform_init_args(self):
        return {}


def transform(train=True, mean=None, std=None, scale=1., size=2000):
    transforms = [
        # alb.PadIfNeeded(
        #     min_height=int(size * scale),
        #     min_width=int(size * scale),
        #     border_mode=cv2.BORDER_CONSTANT,
        # ),
        # DebugAugmentations(),
        alb.Resize(size, size),
        alb.Normalize(mean=_mean, std=_std, max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0)
    ]

    train_transforms = []
    if train:
        train_transforms = [
        ]

    return alb.Compose(train_transforms + transforms, bbox_params=bbox_params)
