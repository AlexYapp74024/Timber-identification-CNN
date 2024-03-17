import torch
from torch import Tensor
from torchvision import transforms
import cv2
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import pandas as pd
# 3264 x 2448

DATA_DIR = "data/image/train"
labels = ["Aspidosperma polyneuron", "Araucaria angustifolia", "Tabebuia sp.", "Cordia goeldiana", "Cordia sp.", "Hura crepitans", "Acrocarpus fraxinifolius", "Hymenaea sp.", "Peltogyne sp.", "Hymenolobium petraeum", "Myroxylon balsamum", "Dipteryx sp.", "Machaerium sp.", "Bowdichia sp.", "Mimosa scabrella", "Cedrelinga catenaeformis", "Goupia glabra", "Ocotea porosa", "Mezilaurus itauba", "Laurus nobilis", "Bertholethia excelsa", "Cariniana estrellensis", "Couratari sp.", "Carapa guianensis", "Cedrela fissilis", "Melia azedarach", "Swietenia macrophylla", "Brosimum paraense", "Bagassa guianensis", "Virola surinamensis", "Eucalyptus sp.", "Pinus sp.", "Podocarpus lambertii", "Grevilea robusta", "Balfourodendron riedelianum", "Euxylophora paraensis", "Micropholis venulosa", "Pouteria pachycarpa", "Manilkara huberi", "Erisma uncinatum", "Vochysia sp."]
label2id = {label:id for id, label in enumerate(labels)}

def compile_image_df(data_dir:str, split_at = 0.9)-> pd.DataFrame:
    dirs = os.listdir(data_dir)
    columns=['Image_ID','Species']
    train = pd.DataFrame(columns=columns)
    val = pd.DataFrame(columns=columns)
    for dir in dirs:
        imgs = [(f"{data_dir}/{dir}/{img}", dir) for img in list(os.listdir(f"{data_dir}/{dir}"))]
        length = len(imgs)
        train_count = int(length * split_at)
        train = pd.concat([train, pd.DataFrame(imgs[:train_count],columns=columns)])
        val = pd.concat([val, pd.DataFrame(imgs[train_count:],columns=columns)])

    return train, val

class TimberDataset(Dataset):
    def __init__(self, 
                 dataframe: pd.DataFrame, 
                 is_train=False, 
                 transform=None,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')) -> None:
        super().__init__()
        self.dataframe = dataframe
        self.is_train = is_train
        self.transform = transform
        self.device = device

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: list[int]|Tensor):
        if torch.is_tensor(idx): 
            idx = idx.tolist()
        
        img_name = os.path.join(self.dataframe.iloc[idx,0])
        image = cv2.imread(img_name)
        image = Image.fromarray(image)

        label = self.dataframe.iloc[idx,1]
        label = label2id[label]
        label = torch.tensor(int(label))

        if self.transform:
            image = self.transform(image)
        return image.to(self.device), label.to(self.device)

def build_dataloader(
        train_ratio = 0.9,
        img_size = (640,640),
        batch_size = 12,
    ) -> tuple[DataLoader,DataLoader]:
    train_df, val_df = compile_image_df(DATA_DIR, split_at=train_ratio)

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(TimberDataset(train_df, is_train=True,transform=transform),
                              shuffle=True,
                              batch_size=batch_size)
    val_loader = DataLoader(TimberDataset(val_df, is_train=True,transform=transform),
                            batch_size=batch_size)
    
    return train_loader,val_loader