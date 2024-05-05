from ultralytics import YOLO

import wandb
wandb.init(project='Bee Detection with YOLOv8', entity='sschnydrig')

# start timer
import time
start_time = time.time()

# Load pretrained yolov8 model
model = YOLO("yolov8l.pt")

#Define the hyperparameters
hyperparameters = {
    "optimizer": "AdamW", 
    "lr0": 0.00772,
    "lrf": 0.00904,
   "momentum": 0.87061,
    "weight_decay": 0.00033,
    "warmup_epochs": 3.14947,
    "warmup_momentum": 0.63019,
    "box": 2.74502,
    "cls": 0.34081,
    "dfl": 0.82888,
    "hsv_h": 0.01784,
    "hsv_s": 0.45962,
    "hsv_v": 0.47245,
    "translate": 0.07826,
    "scale": 0.50594,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.62941,
    "mosaic": 0.94205,
    "mixup": 0.0,
    "copy_paste": 0.0,
}

results = model.train(data="datasets/data.yaml", epochs=50, batch = 16, device = "cuda")  # train the model

# print time it took to train in minutes
print("--- %s minutes ---" % ((time.time() - start_time)/60))