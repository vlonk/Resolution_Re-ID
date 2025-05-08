import sys
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from tqdm import tqdm

from display_data import Market1501Dataset
from resnetmodel   import ReIDResNet50

class SRTrainDataset(Dataset):
    def __init__(self, train_dir, transform=None):
        self.transform = transform
        self.samples = []
        for fname in os.listdir(train_dir):
            if not fname.lower().endswith('.jpg'):
                continue
            pid = int(fname.split('_')[0])
            self.samples.append((os.path.join(train_dir, fname), pid))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, pid = self.samples[idx]
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            raise RuntimeError(f"Failed to load {path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img_rgb).permute(2,0,1).float().div(255.0)
        if self.transform:
            img = self.transform(img)
        return img, pid

# feature extraction & evaluation
def extract_features(loader, model, device):
    model.eval()
    feats, pids = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            emb, _ = model(imgs)
            feats.append(emb.cpu())
            pids.extend(labels)
    return torch.cat(feats, dim=0), pids

def evaluate(query_loader, gallery_loader, model, device):
    import numpy as np
    import torch.nn.functional as F

    q_feats, q_pids = extract_features(query_loader, model, device)
    g_feats, g_pids = extract_features(gallery_loader, model, device)

    sim_mat = torch.mm(
        F.normalize(q_feats, p=2, dim=1),
        F.normalize(g_feats, p=2, dim=1).t()
    ).numpy()

    rank1 = sum(
        1 for i in range(len(q_pids))
        if g_pids[np.argmax(sim_mat[i])] == q_pids[i]
    ) / len(q_pids)

    print(f" → Rank-1 Accuracy: {rank1:.4f}")

# PID remapping
def run_experiment(tag, args):
    print(f"\n===== Running on [{tag}] =====")

    transform = T.Compose([
        T.Resize((256, 128)),
        T.Normalize(mean=[0.485,0.456,0.406],
                    std =[0.229,0.224,0.225]),
    ])

    # build datasets
    if tag == 'sr_x2':
        train_ds   = SRTrainDataset(args.sr_root, transform=transform)
        query_ds   = Market1501Dataset(args.orig_root, mode='query',   transform=transform)
        gallery_ds = Market1501Dataset(args.orig_root, mode='gallery', transform=transform)
    else:
        train_ds   = Market1501Dataset(args.orig_root, mode='train',   transform=transform)
        query_ds   = Market1501Dataset(args.orig_root, mode='query',   transform=transform)
        gallery_ds = Market1501Dataset(args.orig_root, mode='gallery', transform=transform)

    # PID remapping for classification head
    raw_samples = train_ds.samples
    unique_pids = sorted({pid for _, pid in raw_samples})
    pid2label   = {pid: idx for idx, pid in enumerate(unique_pids)}
    # overwrite train_ds.samples with mapped labels
    train_ds.samples = [
        (path, pid2label[pid])
        for path, pid in raw_samples
    ]
    num_ids = len(unique_pids)

    # build dataloaders
    train_loader   = DataLoader(train_ds,   batch_size=args.batch_size, shuffle=True,  num_workers=4)
    query_loader   = DataLoader(query_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4)
    gallery_loader = DataLoader(gallery_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # model, loss, optimizer
    model     = ReIDResNet50(num_ids=num_ids, emb_dim=args.emb_dim).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # training loop
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"[{tag}] Epoch {epoch}/{args.epochs}"):
            imgs, labels = imgs.to(args.device), labels.to(args.device)
            emb, logits  = model(imgs)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[{tag}] Epoch {epoch} — Avg Loss: {avg_loss:.4f}")
        evaluate(query_loader, gallery_loader, model, args.device)

    # save checkpoint
    ckpt_dir = os.path.join('checkpoints', tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'reid_resnet50.pth')
    torch.save(model.state_dict(), ckpt_path)
    print(f"[{tag}] Model saved to {ckpt_path}")

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Train ReID on Market-1501 / ESRGAN outputs")
    p.add_argument('--orig-root',  type=str, required=True, help='original Market-1501 root')
    p.add_argument('--sr-root',    type=str, required=True, help='ESRGAN×2 train folder')
    p.add_argument('--mode',       choices=['orig','sr','both'], default='both')
    p.add_argument('--epochs',     type=int,   default=20)
    p.add_argument('--batch-size', type=int,   default=32)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--emb-dim',    type=int,   default=512)
    p.add_argument('--device',     type=str,   default=('cuda' if torch.cuda.is_available() else 'cpu'))
    return p.parse_args()

if __name__ == '__main__':
    print("▶ train.py running, args=", sys.argv)
    args = parse_args()

    to_run = []
    if args.mode in ('orig','both'):
        to_run.append('original')
    if args.mode in ('sr','both'):
        to_run.append('sr_x2')

    for tag in to_run:
        run_experiment(tag, args)
