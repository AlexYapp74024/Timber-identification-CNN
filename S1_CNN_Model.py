import torchvision
import torch
import os
from torch import nn, Tensor
import torch.nn.functional as F
import cv2
from PIL import Image
import numpy as np

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

class CNN_Model(nn.Module):
    def __init__(self,
                 image_size: tuple[int,int],
                 interpreter: Interpreter,
    ):
        super().__init__()
        self.device = interpreter.device
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

        ratios = preds
        preds = torch.mode(preds, 0).values
        
        return ratios, preds
    
class_count = 41

def build_interpreter(img_size=(640,640), 
                      device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ) -> Interpreter:
    img_size = list(img_size)

    x = torch.randn([3]+img_size).view([-1,3]+img_size).to(device)
        
    return Interpreter(class_count=class_count, sample_yolo_output=x, device=device)

def build_model(img_size = (640,640),
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ) -> CNN_Model:
    return CNN_Model(image_size=img_size, 
                     interpreter=build_interpreter(img_size, device))

if __name__ == "__main__":
    model = build_model(img_size=(320,320))
    DATA_DIR = "data/image/test"
    dir = os.listdir(DATA_DIR)[0]
    img_name = os.listdir(f"{DATA_DIR}/{dir}")[0]
    img_path = f"{DATA_DIR}/{dir}/{img_name}"
    
    out = model.predict_large_image(img_path)
    print(out)