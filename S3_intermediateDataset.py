import os
import torch
from time import time
from torch import Tensor
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from S2_TimberDataset import build_dataloader
from typing import Callable
import pandas as pd
import numpy as np
# 3264 x 2448

def write_random_lowercase(n):
    min_lc = ord(b'a')
    len_lc = 26
    ba = bytearray(os.urandom(n))
    for i, b in enumerate(ba):
        ba[i] = min_lc + b % len_lc # convert 0..255 to 97..122
    return ba.decode("utf-8")

INTERMEDIATE_DIR = "data/intermediate"

class IntermediateDataset(Dataset):
    def __init__(self,
                 name,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')) -> None:
        super().__init__()
        self.name = name
        self.device = device
        files = os.listdir(f"{INTERMEDIATE_DIR}/{self.name}")
        self.tensors = np.array([f for f in files if os.path.splitext(f)[-1] == ".pt"])
        self.labels  = np.array([f for f in files if os.path.splitext(f)[-1] == ".txt"])

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, idx: list[int]|Tensor):
        tensor = self.tensors[idx]
        labels = self.labels[idx]
        images = torch.load(f"{INTERMEDIATE_DIR}/{self.name}/{tensor}")
        
        with open(f"{INTERMEDIATE_DIR}/{self.name}/{labels}", 'r') as f: 
            labels = f.readline().split("-")
            labels = Tensor(list(map(int, labels)))

        images = images.to(self.device) 
        labels = labels.to(device=self.device, dtype=torch.int64)

        return images, labels

    @staticmethod
    def prepare_intermediate_dataset(pred: Callable, name: str, dataset: DataLoader, iterations = 1) -> None:
        with torch.no_grad():
            for _ in range(iterations):
                for images, labels in tqdm(dataset):
                    out = pred(images)
                    labels = np.char.mod('%d', labels.cpu().numpy())
                    labels = '-'.join(labels)

                    file_name = f"{INTERMEDIATE_DIR}/{name}/{int(time())}_{write_random_lowercase(10)}"
                    torch.save(out,f"{file_name}.pt")
                    with open(f"{file_name}.txt", 'w') as f:
                        f.write(labels)

def build_intermediate_dataset_if_not_exists(pred_:Callable, name:str, dataset:DataLoader) -> None:
    try: os.mkdir(INTERMEDIATE_DIR)
    except: pass
    try: os.mkdir(f"{INTERMEDIATE_DIR}/{name}")
    except: pass

    if os.listdir(f"{INTERMEDIATE_DIR}/{name}") == []:
        IntermediateDataset.prepare_intermediate_dataset(pred_, name, dataset)

def intermediate_dataset(name:str) -> DataLoader:
    return DataLoader(IntermediateDataset(name=name),batch_size=1)

if __name__ == '__main__':
    train, val, test = build_dataloader(train_ratio= 0.01)
    
    build_intermediate_dataset_if_not_exists(lambda x:x, "testing", train)

    train_loader = DataLoader(IntermediateDataset("testing"),batch_size=1)
    (i1,i2,i3), val = next(iter(train_loader))
    "a"
    
