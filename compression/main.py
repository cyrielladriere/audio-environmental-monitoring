# https://pytorch.org/blog/torchvision-mobilenet-v3-implementation/
# https://pyimagesearch.com/2021/10/11/pytorch-transfer-learning-and-image-classification/
import pandas as pd
import cv2 as cv
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.quantization import mobilenet_v3_large, MobileNet_V3_Large_QuantizedWeights
from compression.quantization import QAT
from compression.evaluation import print_model_size, evaluate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------- Testing Env
BASE_MODEL = False
QUANTIZATION = True
# ------------- Variables
training_data = "data/train"
val_data = "data/val"
labels = "data/noisy_imagenette.csv"
classes = {"n01440764": 0, "n02102040": 217, "n02979186": 482, "n03000684": 491, "n03028079": 497, "n03394916": 566, "n03417042": 569, "n03425413": 571, "n03445777": 574, "n03888257": 701}
# ------------- Hyperparameters
batch_size = 64

def main():
    train_dataset = ImagenetteDataset(labels, training_data, "train")
    val_dataset = ImagenetteDataset(labels, labels, "val")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    if(BASE_MODEL):
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2).to(device=device)
        model_quantized = mobilenet_v3_large(weights=MobileNet_V3_Large_QuantizedWeights.DEFAULT, quantize=True).to(device=device)
        print_model_size(model)
        print_model_size(model_quantized)

        top1, top5 = evaluate(model, val_dataloader)
        print(f"Evaluation accuracy on {len(val_dataloader)} images: {top1}, {top5}")
    elif(QUANTIZATION):
        qat_model = QAT(train_dataloader, val_dataloader)

        print_model_size(qat_model)
        
        top1, top5 = evaluate(model, val_dataloader)
        print(f"Evaluation accuracy on {len(val_dataloader)} images: {top1}, {top5}")
    return

class ImagenetteDataset(Dataset):
    """Imagenette dataset."""

    def __init__(self, csv_file, root_dir, split, transform=None):
        """
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class
        Args:
            csv_file (string): Path to the csv file with labels.
            root_dir (string): Directory with all the images.
            split (string): Which split is the dataset a part of.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.labels[self.labels['path'].str.contains(f'{self.split}/')])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = f"data/{self.labels.iloc[idx, 0]}"
        image = cv.imread(img_name)
        image = cv.resize(image, (320,320), interpolation= cv.INTER_LINEAR)

        label = self.labels.iloc[idx, 1]
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == "__main__":
    main()