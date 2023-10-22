# https://pytorch.org/blog/torchvision-mobilenet-v3-implementation/
import io
import numpy as np
import pandas as pd
import cv2 as cv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision
import os
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.quantization import mobilenet_v3_large, MobileNet_V3_Large_QuantizedWeights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------- Locations
training_data = "data/train"
val_data = "data/val"
labels = "data/noisy_imagenette.csv"
# ------------- Hyperparameters
train_batch_size = 64
val_batch_size = 64
num_eval_batches = 1000

def main():
    train_dataset = ImagenetteDataset(labels, training_data, "train")
    val_dataset = ImagenetteDataset(labels, labels, "val")

    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True)

    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1).to(device=device)
    model_quantized = mobilenet_v3_large(weights=MobileNet_V3_Large_QuantizedWeights.DEFAULT, quantize=True).to(device=device)
    print_model_size(model)
    print_model_size(model_quantized)

    criterion = nn.CrossEntropyLoss()
    top1, top5 = evaluate(model, criterion, val_dataloader)
    print(f"Evaluation accuracy on {len(val_dataloader)} images: Acc@1={top1}, Acc@5={top5}")

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

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def print_model_size(mdl):
    """Prints the size of the given model in Megabytes"""
    torch.save(mdl.state_dict(), "tmp.pt")
    print("%.2f MB" %(os.path.getsize("tmp.pt")/1e6))
    os.remove('tmp.pt')

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def evaluate(model, criterion, data_loader):#, neval_batches):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for i, sample in enumerate(data_loader):
            image = sample['image'].permute(0, 3, 1, 2)         # change shape: [batch_size, height, width, channels] -> [batch_size, channels, height, width]
            target = sample['label']
            print(image.shape)
            output = model(image)
            # loss = criterion(output, target)
            # cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            print('.', end = '')
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            # if cnt >= neval_batches:
            #      return top1, top5

    return top1, top5

if __name__ == "__main__":
    main()