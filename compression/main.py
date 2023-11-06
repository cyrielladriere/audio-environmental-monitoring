# https://www.kaggle.com/code/daisukelab/cnn-2d-basic-solution-powered-by-fast-ai
import pandas as pd
import numpy as np
import torch
import time
from torch import nn, optim
import os
import random
from PIL import Image
from tempfile import TemporaryDirectory
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.quantization import mobilenet_v3_large, MobileNet_V3_Large_QuantizedWeights
from compression.quantization import QAT
from compression.qat import qat
from compression.evaluation import print_model_size, evaluate
from compression.base_model import MobileNetV2, loss_func
from compression.preprocessing import convert_dataset, load_pkl, save_images, get_labels
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
training_data = "train_curated"
val_data = "test"
model_pth = "resources/MobileNetV2.pth"
# ------------- Hyperparameters
image_size = (128, 64)
batch_size = 64
n_epochs = 20
n_classes = 80
classes = {'Bark': 0, 'Motorcycle': 1, 'Writing': 2, 'Female_speech_and_woman_speaking': 3, 'Tap': 4, 'Child_speech_and_kid_speaking': 5, 'Screaming': 6, 'Meow': 7, 'Scissors': 8, 'Fart': 9, 'Car_passing_by': 10, 'Harmonica': 11, 'Sink_(filling_or_washing)': 12, 'Burping_and_eructation': 13, 'Slam': 14, 'Drawer_open_or_close': 15, 'Cricket': 16, 'Hiss': 17, 'Frying_(food)': 18, 'Sneeze': 19, 'Chink_and_clink': 20, 'Fill_(with_liquid)': 21, 'Crowd': 22, 'Marimba_and_xylophone': 23, 'Sigh': 24, 'Accordion': 25, 'Electric_guitar': 26, 'Cupboard_open_or_close': 27, 'Bicycle_bell': 28, 'Waves_and_surf': 29, 'Stream': 30, 'Bus': 31, 'Toilet_flush': 32, 'Trickle_and_dribble': 33, 'Tick-tock': 34, 'Keys_jangling': 35, 'Acoustic_guitar': 36, 'Finger_snapping': 37, 'Cheering': 38, 'Race_car_and_auto_racing': 39, 'Bass_guitar': 40, 'Yell': 41, 'Water_tap_and_faucet': 42, 'Run': 43, 'Traffic_noise_and_roadway_noise': 44, 'Crackle': 45, 'Skateboard': 46, 'Glockenspiel': 47, 'Computer_keyboard': 48, 'Whispering': 49, 'Zipper_(clothing)': 50, 'Microwave_oven': 51, 'Bathtub_(filling_or_washing)': 52, 'Male_speech_and_man_speaking': 53, 'Gong': 54, 'Shatter': 55, 'Strum': 56, 'Bass_drum': 57, 'Dishes_and_pots_and_pans': 58, 'Accelerating_and_revving_and_vroom': 59, 'Male_singing': 60, 'Gurgling': 61, 'Walk_and_footsteps': 62, 'Printer': 63, 'Cutlery_and_silverware': 64, 'Chirp_and_tweet': 65, 'Clapping': 66, 'Hi-hat': 67, 'Raindrop': 68, 'Gasp': 69, 'Buzz': 70, 'Drip': 71, 'Chewing_and_mastication': 72, 'Squeak': 73, 'Female_singing': 74, 'Church_bell': 75, 'Mechanical_fan': 76, 'Purr': 77, 'Applause': 78, 'Knock': 79}

def main():
    if(PREPROCESSING):
        # convert_dataset(pd.read_csv(training_audio_labels), training_audio_data, training_data) 
        # convert_dataset(pd.read_csv(test_audio_labels), val_audio_data, val_data)
        data = load_pkl(training_data)
        save_images(data, True)
        # print(train["0a9bebde.wav"].shape)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    data = load_pkl(training_data)

    labels = get_labels(data.keys())
    
    train_dataset = TrainDataset(list(data.values()), labels, transform)
    # val_dataset = datasets.ImageFolder(val_data, transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    dataloaders = {"train": train_dataloader}


    if(BASE_MODEL):
        # model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2).to(device=device)
        # model = mobilenet_v3_large(weights=MobileNet_V3_Large_QuantizedWeights.DEFAULT, quantize=True).to(device=device)
        model = MobileNetV2(3200, 1024, 320, 64, 50, 14000, 527).to(device)
        pretrained_weights = torch.load(model_pth)["model"] # keys: {iteration: , model: }
        model.load_state_dict(pretrained_weights, strict=False)
        

        print_model_size(model)
        # print_model_size(model_quantized)


        # Freeze weights
        for param in model.features.parameters():
            param.requires_grad = False

        # model = base_model(model, train_dataloader, val_dataloader, classes, n_epochs)
        model.fc_audioset = nn.Linear(1024, n_classes, bias=True)

        
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) # Decay LR by a factor of 0.1 every 7 epochs
        model.cuda()
        model = train_model(model, dataloaders, optimizer, exp_lr_scheduler, n_epochs)

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

class TrainDataset(Dataset):
    def __init__(self, mels, labels, transforms):
        super().__init__()
        self.mels = mels
        self.labels = labels
        self.transforms = transforms
        
    def __len__(self):
        return len(self.mels)
    
    def __getitem__(self, idx):
        image = Image.fromarray(self.mels[idx], mode='L') # Grayscale
        image = self.transforms(image).div_(255)
        
        label = self.labels[idx]
        # label = torch.tensor([classes[l] for l in label])
        label = np.array([classes[l] for l in label])

        one_hot_vector = np.zeros(len(classes), dtype=int)
        one_hot_vector[label] = 1

        return image, one_hot_vector

def train_model(model, dataloaders, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train']:#, 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device) # Shape: [batch_size, channels, height, width]
                    labels = labels.to(device) # Shape: [batch_size, num_classes]

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)["clipwise_output"] # Ouputs: {"clipwise_output": [batch_size, num_classes], "Embedding": }

                        _, preds = torch.max(outputs, 1)    # Shape preds: [batch_size]
                        loss = loss_func(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += 0 # torch.sum(preds == labels.data) # Shape labels.data: [batch_size, num_classes]
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / len(dataloaders[phase])
                epoch_acc = 0 #running_corrects.double() / len(dataloaders[phase])

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model


if __name__ == "__main__":
    main()