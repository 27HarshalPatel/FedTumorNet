"""Preprocessing transforms for Brain Tumor MRI classification."""
import torchvision.transforms as T

def get_train_transforms(image_size=224):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_eval_transforms(image_size=224):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_inverse_normalize():
    return T.Compose([
        T.Normalize(mean=[0.,0.,0.], std=[1/0.229, 1/0.224, 1/0.225]),
        T.Normalize(mean=[-0.485,-0.456,-0.406], std=[1.,1.,1.]),
    ])
