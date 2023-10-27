# https://pytorch.org/blog/torchvision-mobilenet-v3-implementation/
# https://pyimagesearch.com/2021/10/11/pytorch-transfer-learning-and-image-classification/
import pandas as pd
import cv2 as cv
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.quantization import mobilenet_v3_large, MobileNet_V3_Large_QuantizedWeights
from compression.quantization import QAT
from compression.qat import qat
from compression.evaluation import print_model_size, evaluate
from compression.base_model import base_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------- Testing Env
BASE_MODEL = True
QUANTIZATION = False
PRUNING = False
# ------------- Variables
training_data = "data/train"
val_data = "data/val"
labels = "data/noisy_imagenette.csv"
# ------------- Hyperparameters
image_size = (224, 224)
batch_size = 64
n_epochs = 20

def main():
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(training_data, transform)
    val_dataset = datasets.ImageFolder(val_data, transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    classes = train_dataset.classes

    if(BASE_MODEL):
        # model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2).to(device=device)
        model= mobilenet_v3_large(weights=MobileNet_V3_Large_QuantizedWeights.DEFAULT, quantize=True).to(device=device)
        # print_model_size(model)
        # print_model_size(model_quantized)
        model = base_model(model, train_dataloader, val_dataloader, classes, n_epochs)

        top1, top5, inference_time = evaluate(model, val_dataloader, classes)
        print(f"Evaluation accuracy on {len(val_dataset)} images: {top1}, {top5}")
        print("Average inference time: %.4fs" %(inference_time/len(val_dataset)))
    elif(QUANTIZATION):
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2).to(device=device)
        # qat_model = QAT(train_dataloader, val_dataloader)
        # qat_model = qat(train_dataloader, val_dataloader)
        # qat_model = mbv2(model, train_dataloader)

        # print_model_size(qat_model)
        
        top1, top5 = evaluate(model, val_dataloader)
        print(f"Evaluation accuracy on {len(val_dataset)} images: {top1}, {top5}")
    return


if __name__ == "__main__":
    main()