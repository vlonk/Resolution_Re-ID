import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class Market1501Dataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        """
        root_dir: path to Market-1501 root
        mode: 'train', 'query', or 'gallery'
        transform: torchvision transforms to apply
        """
        assert mode in ('train','query','gallery')
        folder = {
            'train': 'bounding_box_train',
            'query': 'query',
            'gallery': 'bounding_box_test'
        }[mode]
        data_dir = os.path.join(root_dir, folder)
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"{data_dir} not found")

        self.transform = transform
        self.samples = []

        for fname in os.listdir(data_dir):
            if not fname.lower().endswith('.jpg'):
                continue
            pid = int(fname.split('_')[0])
            path = os.path.join(data_dir, fname)
            self.samples.append((path, pid))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, pid = self.samples[idx]
        # load with cv2 (BGR → RGB)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read {img_path}")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2,0,1)  # [H,W,3] → [3,H,W]
        img = img.float().div(255.0)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid

transform = T.Compose([
    T.Resize((256,128)),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

train_ds = Market1501Dataset('Market-1501-v15.09.15', mode='train', transform=transform)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)

