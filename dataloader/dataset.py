import os
import random
from typing import List, Tuple, Dict

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _is_image(fn: str) -> bool:
    return os.path.splitext(fn)[1].lower() in IMG_EXTS


def list_cases(split_dir: str) -> Tuple[List[str], List[int], List[str]]:
    class_names = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
    class_names.sort()
    cls2id = {c: i for i, c in enumerate(class_names)}

    case_dirs, labels = [], []
    for c in class_names:
        cdir = os.path.join(split_dir, c)
        for case in os.listdir(cdir):
            case_dir = os.path.join(cdir, case)
            if not os.path.isdir(case_dir):
                continue
            has_img = any(_is_image(fn) for fn in os.listdir(case_dir))
            if has_img:
                case_dirs.append(case_dir)
                labels.append(cls2id[c])
    return case_dirs, labels, class_names




# Training set image transformations
train_transform = transforms.Compose([transforms.RandomResizedCrop((224,224), scale=(0.4, 1)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ColorJitter(brightness=0.4, contrast=0.4),
                                    transforms.RandomAffine(degrees=15, scale=(0.8,1.2)),
                                    transforms.RandomGrayscale(p=0.2),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])

# Validation set image transformations
val_transform = transforms.Compose([transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])




class CaseFolderDataset(Dataset):
    # Each item is a case folder (bag) containing variable number of images.
    # - If case has > max_images: randomly sample max_images images.
    # - If case has < max_images: keep as is (NO image-level padding).
    #   Feature padding to max_images happens inside the model.
    def __init__(
        self,
        root_dir: str,
        split: str,
        max_images: int = 32,
        augment: bool = False,
    ):
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)
        assert os.path.isdir(self.split_dir), f"Split dir not found: {self.split_dir}"

        self.case_dirs, self.labels, self.class_names = list_cases(self.split_dir)
        self.max_images = max_images

        if augment:
            self.transform = train_transform
        else:
            self.transform = val_transform

    def __len__(self) -> int:
        return len(self.case_dirs)

    def __getitem__(self, idx: int) -> Dict:
        case_dir = self.case_dirs[idx]
        label = self.labels[idx]
        rel = os.path.relpath(case_dir, self.split_dir)
        img_names = os.listdir(case_dir)
        random.shuffle(img_names)
        
        imgs = []
        img_paths = []
        for img_name in img_names[:self.max_images]:
            img_path = os.path.join(case_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            imgs.append(img)
            img_paths.append(img_path)
            
        imgs = torch.stack(imgs)    
        return {"images": imgs, "label": label, "case_id": img_paths}



def collate_cases_no_image_pad(batch: List[Dict]) -> Dict:
    # Collate WITHOUT image-level padding.
    images_list = [b["images"] for b in batch]  # list[(n_i,3,H,W)]
    lengths = [t.shape[0] for t in images_list]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    case_ids = [b["case_id"] for b in batch]
    images = torch.cat(images_list, dim=0)
    
    return {"images": images, "lengths": lengths, "labels": labels, "case_ids": case_ids}


