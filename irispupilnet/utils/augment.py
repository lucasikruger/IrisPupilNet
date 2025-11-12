import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(img_size: int, mode: str, num_channels: int = 3):
    mean = tuple([0.0] * num_channels)
    std = tuple([1.0] * num_channels)
    transforms = [A.Resize(img_size, img_size)]

    if mode == "train":
        transforms.extend([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(0.1, 0.1, p=0.5),
            A.GaussianBlur(blur_limit=(3,3), p=0.2),
        ])

    transforms.extend([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(transpose_mask=True),
    ])
    return A.Compose(transforms)
