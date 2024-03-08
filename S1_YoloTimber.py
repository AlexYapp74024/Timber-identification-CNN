import torchvision
import torch
import hubconf
import os
from torch import nn, Tensor
import torch.nn.functional as F
import cv2
from PIL import Image
import numpy as np

model_name = "yolov6s"

from yolov6.models.yolo import Model as YoloModel
from yolov6.utils.config import Config
config = Config.fromfile(f"configs/base/{model_name}_base_finetune.py")
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class YoloBackbone(YoloModel):
    def __init__(self, config, num_classes, device):
        super().__init__(config, num_classes=num_classes)
        
        self.to(device)
        self.train()

    def forward(self, x:Tensor) -> Tensor:
        # x = self.backbone.forward(x)
        # x = self.neck.forward(x)
        # x = self.detect.forward(x)
        # _,_,_,x = x
        return x

class Interpreter(nn.Module):
    def __init__(self, 
                 class_count:int,
                 sample_yolo_output,
                 device,
                ):
        super().__init__()

        c = 32

        self.train()
        self._conv1 = nn.Conv2d(in_channels= 3,   out_channels= 2*c,  kernel_size=5, padding=2)
        self._conv2 = nn.Conv2d(in_channels= 2*c, out_channels= 4*c,  kernel_size=5, padding=2)
        self._conv3 = nn.Conv2d(in_channels= 4*c, out_channels= 8*c,  kernel_size=5, padding=2)
        self._conv4 = nn.Conv2d(in_channels= 8*c, out_channels=16*c,  kernel_size=3, padding=1)
        self._conv5 = nn.Conv2d(in_channels=16*c, out_channels=32*c,  kernel_size=3, padding=1)
        self._conv6 = nn.Conv2d(in_channels=32*c, out_channels=64*c,  kernel_size=3, padding=1)

        self._linear_size = self.calc_linear(sample_yolo_output)
        print(self._linear_size)

        self._fc1 = nn.Linear(self._linear_size,512)
        self._fc2 = nn.Linear(512, class_count)
        
        self.to(device)
        self.device = device
        self.training = True
        self.train()

    def calc_linear(self, sample_yolo_output) -> int:
        x = self.convs(sample_yolo_output.to('cpu'))
        return x.shape[-1]

    def convs(self, x:Tensor) -> Tensor:
        x = F.max_pool2d(F.relu(self._conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self._conv2(x)), (2,2))        
        x = F.max_pool2d(F.relu(self._conv3(x)), (2,2))
        x = F.max_pool2d(F.relu(self._conv4(x)), (2,2))
        x = F.max_pool2d(F.relu(self._conv5(x)), (2,2))
        x = F.max_pool2d(F.relu(self._conv6(x)), (2,2))
        x = torch.flatten(x,1)
        return x
    
    def fc(self, x:Tensor) -> Tensor:
        x = F.relu(self._fc1(x))
        # x = F.relu(self._fc2(x))
        x = self._fc2(x)
        return x

    def forward(self, x:list[Tensor]) -> Tensor:
        x = self.convs(x)
        x = self.fc(x)
        return x

import patchify
from torchvision import transforms

class YoloTimber(nn.Module):
    def __init__(self,
                 image_size: tuple[int,int],
                 yolo_model: YoloBackbone,
                 interpreter: Interpreter,
    ):
        super().__init__()
        self.device = interpreter.device
        self.yolo_model = yolo_model
        self.image_size = image_size
        self.interpreter = interpreter

    def predict(self, img_path:str) -> Tensor:
        img = cv2.imread(img_path)
        img = Image.fromarray(img)
        img = transforms.ToTensor()(img)
        img = torchvision.transforms.Resize(self.image_size)(img)
        img = img[None]
        img = img.to(self.device)

        preds = self.forward(img)
        _, preds = torch.max(preds,1)
        return preds
    
    def forward(self, x:Tensor) -> Tensor:
        x = self.yolo_model(x)
        x = self.interpreter(x)
        return x

    def predict_large_image(self, 
                   img: np.ndarray,
                   patch_size:int = 816,
        ) -> Tensor:
        
        L = patch_size
        patches = patchify.patchify(img,(L,L,3),L)
        w,h,_ = patches.shape[:3]
        patches = patches.reshape(w*h,*patches.shape[3:]).transpose((0,3,1,2))

        patches = torch.from_numpy(patches)

        patches = patches.float() / 255
        patches = transforms.Resize(self.image_size)(patches)
        patches = patches.to(self.device)

        preds = self.forward(patches)
        _, preds = torch.max(preds,1)
        preds = torch.mode(preds, 0).values
        return preds
    
class_count = 41

def build_backbone(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')) -> YoloBackbone:
    return YoloBackbone(
        config = config,
        num_classes=class_count,
        device = device
    )

def build_interpreter(img_size=(640,640), 
                      yolo_model = None,
                      device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ) -> Interpreter:
    img_size = list(img_size)
    if yolo_model == None:
        yolo_model = build_backbone(device)

    x = torch.randn([3]+img_size).view([-1,3]+img_size).to(device)
    x = yolo_model(x)
        
    return Interpreter(class_count=class_count, sample_yolo_output=x, device=device)

def build_model(img_size = (640,640),
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ) -> YoloTimber:
    yolo_model=build_backbone(device)
    return YoloTimber(yolo_model=yolo_model,
                      image_size=img_size,
                      interpreter=build_interpreter(img_size, yolo_model, device))

if __name__ == "__main__":
    model = build_model(img_size=(320,320))
    DATA_DIR = "data/image/test"
    dir = os.listdir(DATA_DIR)[0]
    img_name = os.listdir(f"{DATA_DIR}/{dir}")[0]
    img_path = f"{DATA_DIR}/{dir}/{img_name}"
    
    out = model.predict_large_image(img_path)
    print(out)