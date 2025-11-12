import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(img_size: int, mode: str):
    if mode == "train":
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(0.1, 0.1, p=0.5),
            A.GaussianBlur(blur_limit=(3,3), p=0.2),
            A.Normalize(mean=(0,0,0), std=(1,1,1)),
            ToTensorV2(transpose_mask=True),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0,0,0), std=(1,1,1)),
            ToTensorV2(transpose_mask=True),
        ])
