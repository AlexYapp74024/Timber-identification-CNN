# Download files
from zipfile import ZipFile
import tarfile
import requests
from tqdm import tqdm
import os
from joblib import Parallel, delayed

def listdir_full(path: str) -> list[str]:
    return [f"{path}/{p}" for p in os.listdir(path)]

def download_file(url, pos):
    local_filename = f"data/{url.split('/')[-1]}"
    if not os.path.exists(local_filename):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()

            total_size = int(r.headers.get("content-length", 0))
            block_size = 8192
            with tqdm(total=total_size,unit="B", unit_scale=True, position=pos, desc=f"#{pos} {local_filename}", ncols=100) as p_bar:
                with open(local_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=block_size): 
                        p_bar.update(len(chunk))
                        f.write(chunk)
    return local_filename

IMAGE_DIR = "data/image"
if not os.path.isdir(IMAGE_DIR):
    links = ["http://www.inf.ufpr.br/lesoliveira/download/macroscopic0.zip",
             "http://www.inf.ufpr.br/lesoliveira/download/macroscopic1.tar.gz",
             "http://www.inf.ufpr.br/lesoliveira/download/macroscopic2.tar.gz",
             "http://www.inf.ufpr.br/lesoliveira/download/macroscopic3.tar.gz",
             "http://www.inf.ufpr.br/lesoliveira/download/macroscopic4.tar.gz",
    ]
    archives : list[str] = Parallel(-1)(delayed(download_file)(l, i) for i , l in enumerate(links))
    
    def unzip(file: str):
        if file.endswith(".zip"):
            with ZipFile(file) as zip_file: zip_file.extractall(IMAGE_DIR)
        if file.endswith(".tar.gz"):
            with tarfile.open(file, "r:gz") as tar: tar.extractall(IMAGE_DIR)
    Parallel(-1)(delayed(unzip)(file) for file in archives)
    # delete faulty images
    [os.remove(f"{IMAGE_DIR}/{i}") for i in os.listdir(IMAGE_DIR) if i.startswith("._")]

from pathlib import Path
import shutil

images = os.listdir(IMAGE_DIR)
# Group by species
labels = ["Aspidosperma polyneuron", "Araucaria angustifolia", "Tabebuia sp.", "Cordia goeldiana", "Cordia sp.", "Hura crepitans", "Acrocarpus fraxinifolius", "Hymenaea sp.", "Peltogyne sp.", "Hymenolobium petraeum", "Myroxylon balsamum", "Dipteryx sp.", "Machaerium sp.", "Bowdichia sp.", "Mimosa scabrella", "Cedrelinga catenaeformis", "Goupia glabra", "Ocotea porosa", "Mezilaurus itauba", "Laurus nobilis", "Bertholethia excelsa", "Cariniana estrellensis", "Couratari sp.", "Carapa guianensis", "Cedrela fissilis", "Melia azedarach", "Swietenia macrophylla", "Brosimum paraense", "Bagassa guianensis", "Virola surinamensis", "Eucalyptus sp.", "Pinus sp.", "Podocarpus lambertii", "Grevilea robusta", "Balfourodendron riedelianum", "Euxylophora paraensis", "Micropholis venulosa", "Pouteria pachycarpa", "Manilkara huberi", "Erisma uncinatum", "Vochysia sp."]
def group_label(id:int, label:str):
    id = f"{id+1:02d}"
    class_dir = f"{IMAGE_DIR}/{label}"
    Path(class_dir).mkdir(parents=True, exist_ok=True)
    
    imgs = [im for im in images if im.startswith(id)]
    [shutil.move(f"{IMAGE_DIR}/{im}",f"{class_dir}/{im}") for im in imgs]

Parallel(-1)(delayed(group_label)(i,l) for i,l in enumerate(labels))

# Train, Test Split
train_dir = f"{IMAGE_DIR}/train"
test_dir = f"{IMAGE_DIR}/test"
test_full_dir = f"{IMAGE_DIR}/test_full"

Path(train_dir).mkdir(parents=True, exist_ok=True)
Path(test_dir).mkdir(parents=True, exist_ok=True)
Path(test_full_dir).mkdir(parents=True, exist_ok=True)

train_ratio = 0.9
dirs = os.listdir(IMAGE_DIR)

def train_test_split(dir:str):
    imgs = os.listdir(f"{IMAGE_DIR}/{dir}")
    split_index = int(len(imgs) * train_ratio)
    train, test = imgs[:split_index], imgs[split_index:]
    Path(f"{train_dir}/{dir}").mkdir(parents=True, exist_ok=True)
    Path(f"{test_dir}/{dir}").mkdir(parents=True, exist_ok=True)
    Path(f"{test_full_dir}/{dir}").mkdir(parents=True, exist_ok=True)

    [shutil.move(f"{IMAGE_DIR}/{dir}/{t}",f"{train_dir}/{dir}/{t}") for t in train]
    [shutil.copy(f"{IMAGE_DIR}/{dir}/{t}",f"{test_full_dir}/{dir}/{t}") for t in test]
    [shutil.move(f"{IMAGE_DIR}/{dir}/{t}",f"{test_dir}/{dir}/{t}") for t in test]
    shutil.rmtree(f"{IMAGE_DIR}/{dir}")

[train_test_split(d) for d in dirs if d not in ["test", "train", "test_full"]]

# Split to patches
import cv2
import patchify

L = 816
# tiles = [im[x:x+M,y:y+N] for x in range(0,im.shape[0],M) for y in range(0,im.shape[1],N)]
dirs = os.listdir(train_dir)
def patch_dir(dir):
    for img in os.listdir(dir):
        path = f"{dir}/{img}"
        img = cv2.imread(f"{dir}/{img}")
        os.remove(path)
        patches = patchify.patchify(img,(L,L,3),L)
        w,h,_ = patches.shape[:3]
        patches = patches.reshape(w*h,*patches.shape[3:])
        path, ext = os.path.splitext(path)
        for i, p in enumerate(patches):
            cv2.imwrite(f"{path}_{i}{ext}",p)

Parallel(-1)(delayed(patch_dir)(d) for d in listdir_full(train_dir))
Parallel(-1)(delayed(patch_dir)(d) for d in listdir_full(test_dir ))