# https://www.kaggle.com/code/daisukelab/cnn-2d-basic-solution-powered-by-fast-ai
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
from compression.base_model import MobileNetV2
from compression.preprocessing import convert_dataset, load_pkl
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------- Testing Env
PREPROCESSING = False
BASE_MODEL = True
QUANTIZATION = False
PRUNING = False
# ------------- Variables
training_audio_data = "data/audio/train_curated"
val_audio_data = "data/audio/test"
training_audio_labels = "data/audio/train_curated.csv"
test_audio_labels = "data/audio/sample_submission.csv"
training_data = "data/images/train_curated"
val_data = "data/images/test"
model_pth = "resources/MobileNetV2.pth"
# ------------- Hyperparameters
image_size = (224, 224)
batch_size = 64
n_epochs = 20

def main():
    if(PREPROCESSING):
        convert_dataset(pd.read_csv(training_audio_labels), training_audio_data, training_data) 
        convert_dataset(pd.read_csv(test_audio_labels), val_audio_data, val_data)
        # train = load_pkl(training_data)
        # print(train["0a9bebde.wav"].shape)
        return
    # transform = transforms.Compose([
    #     transforms.Resize(image_size),
    #     transforms.ToTensor()
    # ])

    # train_dataset = datasets.ImageFolder(training_data, transform)
    # val_dataset = datasets.ImageFolder(val_data, transform)

    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    # classes = train_dataset.classes

    if(BASE_MODEL):
        # model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2).to(device=device)
        # model = mobilenet_v3_large(weights=MobileNet_V3_Large_QuantizedWeights.DEFAULT, quantize=True).to(device=device)
        model = MobileNetV2(32000, 1024, 320, 64, 50, 14000, 527).to(device)
        pretrained_weights = torch.load(model_pth, map_location=device)["model"] # keys: {iteration: , model: }
        model.load_state_dict(pretrained_weights)

        
        print_model_size(model)
        # print_model_size(model_quantized)

        # model = base_model(model, train_dataloader, val_dataloader, classes, n_epochs)

        # top1, top5, inference_time = evaluate(model, val_dataloader, classes)
        # print(f"Evaluation accuracy on {len(val_dataset)} images: {top1}, {top5}")
        # print("Average inference time: %.4fs" %(inference_time/len(val_dataset)))

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