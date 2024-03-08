import gdown
import os
import torch
from S1_CNN_Model import CNN_Model
import gradio as gr
import numpy as np
import cv2
import pandas as pd

MODEL_LINK = "https://drive.google.com/file/d/1YUTQlIjNEW8w7Y1kXCOlBNgrgTmdQBoI/view?usp=sharing"
MODEL_PATH = "model.pt"

if not os.path.exists(MODEL_PATH):
    print("Downloading model . . . ")
    gdown.download(MODEL_LINK,MODEL_PATH,fuzzy=True)

model:CNN_Model = torch.load(MODEL_PATH)

def listdir_full(path: str) -> list[str]:
    return [f"{path}/{p}" for p in os.listdir(path)]

SAMPLE_DIR = "data/image/test_full"
labels = os.listdir(SAMPLE_DIR)

class History():
    cols = ["Image", "Prediction"]

    def __init__(self, img, name) -> None:
        self.img = resize_image(img)
        self.name = name

MAX_IMG_LEN = 160
def resize_image(img):
    h, w, _ = img.shape

    if w > h: 
        w1 = MAX_IMG_LEN
        h1 = int(h/w * MAX_IMG_LEN) 
    else:
        h1 = MAX_IMG_LEN
        w1 = int(w/h * MAX_IMG_LEN)
    return cv2.resize(img,(w1,h1))

PD_COLS=["image","predicted species"]
MAX_HISTORY = 10

def classify(image: np.array, history):
    if history == None: history = []

    with torch.no_grad():
        pred = model.predict_large_image(cv2.cvtColor(image, cv2. COLOR_RGB2BGR)).item()
        pred = labels[pred]

    history += [(resize_image(image), pred)] 
    hist = history[-MAX_HISTORY:]

    return pred, *toggle_history_components(hist), history

def toggle_history_components(history: list[History]):
    n_hidden = MAX_HISTORY - len(history)
    images, names = list(zip(*history))

    components = [gr.Image(x, visible=True) for x in images]
    components += [gr.Image(visible=False)] * n_hidden
    components += [gr.Markdown(x, visible=True) for x in names]
    components += [gr.Markdown(visible=False)] * n_hidden
    return components

def classification_tab():
    with gr.Row():
        with gr.Column():
            image = gr.Image()
            with gr.Row():
                submit = gr.Button("Submit", variant='primary')
                clear  = gr.ClearButton(image)
        pred = gr.Textbox(label="Prediction")
    
    return image, submit, clear, pred

MAX_SAMPLE_COUNT = max([len(os.listdir(x)) for x in listdir_full(SAMPLE_DIR)])

def sample_tab(image_input, tabs):
    
    def choose_image(image):
        return gr.Image(image), gr.Tabs(selected=0)
    
    def refresh_samples(species):
        images = listdir_full(f"{SAMPLE_DIR}/{species}")
        n_hidden = MAX_SAMPLE_COUNT-len(images)
        
        components = [gr.Image(i,visible=True) for i in images]
        components += [gr.Image(visible=False)] * n_hidden
        components += [gr.Button(visible=True) for _ in images]
        components += [gr.Button(visible=False)] * n_hidden
        return components

    dropdown = gr.Dropdown(labels, label="Species", value="Select a Species")

    images = []
    buttons = []

    def sample_panel():
      with gr.Column():
        image = gr.Image(visible=False ,interactive=False, min_width=1)
        select = gr.Button("Submit", variant='primary', visible=False)
        
        images.append(image)
        buttons.append(select)
        select.click(choose_image, image, [image_input, tabs])

    with gr.Row(): [sample_panel() for _ in range(MAX_SAMPLE_COUNT)]
    
    dropdown.change(refresh_samples, dropdown, images+buttons)
    return 

def history_tab():
    history_imgs = []
    history_names = []
    with gr.Row():
        gr.Markdown("# Image")
        gr.Markdown("# Species")
        gr.Markdown("")
    
    with gr.Column():
      for _ in range(MAX_HISTORY):
        with gr.Row():
            history_imgs.append(gr.Image(height=200,visible=False))
            history_names.append(gr.Markdown("A",visible=False))
            gr.Markdown("")

    return history_imgs + history_names

with gr.Blocks() as demo:
    history = gr.State([])
    with gr.Tabs() as tabs:
        with gr.Tab("Classification", id=0):
            image, submit, clear, pred = classification_tab()
        
        with gr.Tab("Samples", id=1):
            sample_tab(image, tabs)
        
        with gr.Tab("History", id=2): 
            table_contents = history_tab()

            # history = gr.Gallery(interactive=False)
    submit.click(classify,[image, history],[pred, *table_contents, history])

demo.launch()
