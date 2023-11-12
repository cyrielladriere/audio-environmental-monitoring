# https://www.kaggle.com/code/daisukelab/cnn-2d-basic-solution-powered-by-fast-ai
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import time
from torchmetrics.classification import BinaryAccuracy, BinaryConfusionMatrix
from torch import nn, optim
import os
import random
from PIL import Image
from tempfile import TemporaryDirectory
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from compression.evaluation import print_model_size, evaluate
from compression.models.PANN_pretrained import MobileNetV2, loss_func
from compression.models.base_model import MobileNetV3, _mobilenet_v3_conf
from compression.models.AT_pretrained import MN, _mn_conf
from compression.preprocessing import convert_dataset, load_pkl, save_images, get_labels, convert_labels
from compression.models.quantized_model import QuantizableMobileNetV3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------- Testing Env
PREPROCESSING = False
BASE_MODEL = False
MODEL_AT = True
MODEL_PANN = False
QUANTIZATION = False
PRUNING = False
# ------------- Variables
training_audio_data = "data/audio/train_curated"
val_audio_data = "data/audio/test"
training_audio_labels = "data/audio/train_curated.csv"
test_audio_labels = "data/audio/sample_submission.csv"
training_data = "data/train_curated"
val_data = "data/test"
model_pann = "resources/MobileNetV2.pth"
model_at = "resources/mn10_as.pt"
# ------------- Hyperparameters
image_size = (256, 128)
batch_size = 64
n_epochs = 200
n_classes = 80
classes = {'Bark': 0, 'Motorcycle': 1, 'Writing': 2, 'Female_speech_and_woman_speaking': 3, 'Tap': 4, 'Child_speech_and_kid_speaking': 5, 'Screaming': 6, 'Meow': 7, 'Scissors': 8, 'Fart': 9, 'Car_passing_by': 10, 'Harmonica': 11, 'Sink_(filling_or_washing)': 12, 'Burping_and_eructation': 13, 'Slam': 14, 'Drawer_open_or_close': 15, 'Cricket': 16, 'Hiss': 17, 'Frying_(food)': 18, 'Sneeze': 19, 'Chink_and_clink': 20, 'Fill_(with_liquid)': 21, 'Crowd': 22, 'Marimba_and_xylophone': 23, 'Sigh': 24, 'Accordion': 25, 'Electric_guitar': 26, 'Cupboard_open_or_close': 27, 'Bicycle_bell': 28, 'Waves_and_surf': 29, 'Stream': 30, 'Bus': 31, 'Toilet_flush': 32, 'Trickle_and_dribble': 33, 'Tick-tock': 34, 'Keys_jangling': 35, 'Acoustic_guitar': 36, 'Finger_snapping': 37, 'Cheering': 38, 'Race_car_and_auto_racing': 39, 'Bass_guitar': 40, 'Yell': 41, 'Water_tap_and_faucet': 42, 'Run': 43, 'Traffic_noise_and_roadway_noise': 44, 'Crackle': 45, 'Skateboard': 46, 'Glockenspiel': 47, 'Computer_keyboard': 48, 'Whispering': 49, 'Zipper_(clothing)': 50, 'Microwave_oven': 51, 'Bathtub_(filling_or_washing)': 52, 'Male_speech_and_man_speaking': 53, 'Gong': 54, 'Shatter': 55, 'Strum': 56, 'Bass_drum': 57, 'Dishes_and_pots_and_pans': 58, 'Accelerating_and_revving_and_vroom': 59, 'Male_singing': 60, 'Gurgling': 61, 'Walk_and_footsteps': 62, 'Printer': 63, 'Cutlery_and_silverware': 64, 'Chirp_and_tweet': 65, 'Clapping': 66, 'Hi-hat': 67, 'Raindrop': 68, 'Gasp': 69, 'Buzz': 70, 'Drip': 71, 'Chewing_and_mastication': 72, 'Squeak': 73, 'Female_singing': 74, 'Church_bell': 75, 'Mechanical_fan': 76, 'Purr': 77, 'Applause': 78, 'Knock': 79}

def main():
    if(PREPROCESSING):
        convert_dataset(pd.read_csv(training_audio_labels), training_audio_data, training_data) 
        # convert_dataset(pd.read_csv(test_audio_labels), val_audio_data, val_data)
        # data = load_pkl(training_data)
        # save_images(data, True)
        # print(train["0a9bebde.wav"].shape)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    data = load_pkl(training_data)

    labels = get_labels(data.keys())

    labels = convert_labels(labels)

    global y_val, x_val

    x_trn, x_val, y_trn, y_val = train_test_split(list(data.values()), labels, test_size=0.2)
    
    train_dataset = TrainDataset(x_trn, y_trn, transform)
    val_dataset = TrainDataset(x_val, y_val, transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    dataloaders = {"train": train_dataloader, "val": val_dataloader}

    if(BASE_MODEL):
        inverted_residual_setting, last_channel = _mobilenet_v3_conf()
        model = MobileNetV3(inverted_residual_setting=inverted_residual_setting, last_channel=last_channel, num_classes=80).to(device)

        print_model_size(model)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        exp_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
        model.cuda()
        model = train_model(model, dataloaders, optimizer, exp_lr_scheduler, n_epochs)
    elif(MODEL_AT):
        inverted_residual_setting, last_channel = _mn_conf()
        model = MN(inverted_residual_setting=inverted_residual_setting, last_channel=last_channel, num_classes=527).to(device)
        pretrained_weights = torch.load(model_at)
        model.load_state_dict(pretrained_weights, strict=False)

        print_model_size(model)

        # Freeze weights
        for param in model.features.parameters():
            param.requires_grad = False

        # Initialize layers that are not frozen
        model.classifier[2] = nn.Linear(in_features=960, out_features=1280, bias=True)   
        model.classifier[5] = nn.Linear(1280, n_classes, bias=True)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        exp_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
        model.cuda()
        model = train_model(model, dataloaders, optimizer, exp_lr_scheduler, n_epochs)
    elif(MODEL_PANN):
        model = MobileNetV2(44100, 1024, 320, 64, 50, 14000, 527).to(device)
        pretrained_weights = torch.load(model_pann)["model"] # keys: {iteration: , model: }
        model.load_state_dict(pretrained_weights)

        print_model_size(model)

        # Freeze weights
        for param in model.features.parameters():
            param.requires_grad = False

        # Initialize layers that are not frozen
        model.bn0 = nn.BatchNorm2d(128)
        model.fc1 = nn.Linear(in_features=1280, out_features=256, bias=True)    # out_features tested: 1024(pretty bad), 512(ok), 128(okok)
        model.fc_audioset = nn.Linear(256, n_classes, bias=True)
        
       
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        exp_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
        model.cuda()
        model = train_model(model, dataloaders, optimizer, exp_lr_scheduler, n_epochs)

        # top1, top5, inference_time = evaluate(model, val_dataloader, classes)
        # print(f"Evaluation accuracy on {len(val_dataset)} images: {top1}, {top5}")
        # print("Average inference time: %.4fs" %(inference_time/len(val_dataset)))

    elif(QUANTIZATION):
        inverted_residual_setting, last_channel = _mobilenet_v3_conf()
        qat_model = QuantizableMobileNetV3(inverted_residual_setting=inverted_residual_setting, last_channel=last_channel, num_classes=80).to(device)

        print_model_size(qat_model)

        qat_model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')
        torch.ao.quantization.prepare_qat(qat_model, inplace=True)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        exp_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
        qat_model.cuda()
        qat_model = train_model(qat_model, dataloaders, optimizer, exp_lr_scheduler, n_epochs)

        qat_model.to("cpu") # Needed for quatization convert
        model_qat = torch.quantization.convert(qat_model.eval(), inplace=False)

        print_model_size(qat_model)
        # top1, top5 = evaluate(model, val_dataloader)
        # print(f"Evaluation accuracy on {len(val_dataset)} images: {top1}, {top5}")
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
        # image = Image.fromarray(self.mels[idx], mode='RGB') # RGB
        image = self.transforms(image).div_(255)
        
        label = self.labels[idx]
        # label = torch.tensor([classes[l] for l in label])
        # label = np.array([classes[l] for l in label])

        one_hot_vector = np.zeros(len(classes), dtype=int)
        one_hot_vector[label] = 1

        return image, one_hot_vector

def train_model(model, dataloaders, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_lwlrap = 0.0
        best_epoch = 0

        for epoch in range(1, num_epochs+1):
            print(f'Epoch {epoch}/{num_epochs}')
            print('-' * 10)

            valid_preds = np.zeros((len(x_val), len(classes)))

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                avg_loss = 0.0
                running_loss = 0.0
                running_corrects = 0

                TN = 0 
                FP = 0 
                FN = 0 
                TP = 0 

                # Iterate over data.
                for i, data in enumerate(dataloaders[phase]):
                    inputs, labels = data
                    inputs = inputs.to(device) # Shape: [batch_size, channels, height, width]
                    labels = labels.to(device) # Shape: [batch_size, num_classes]

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    metric = BinaryAccuracy(threshold=0.5).to(device)
                    bcm = BinaryConfusionMatrix(threshold=0.5).to(device)
                    
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        if MODEL_PANN:
                            # Outputs: {"clipwise_output": [batch_size, num_classes], "Embedding": }
                            outputs = model(inputs)["clipwise_output"]
                        elif MODEL_AT:
                            outputs = model(inputs)[0]
                        else:
                            outputs = model(inputs)


                        # if(epoch == 5 or epoch == 10 or epoch == 1):
                        #     print(torch.max(model.fc1.weight))

                        # loss = loss_func(outputs, labels)

                        criterion = nn.BCEWithLogitsLoss() # With logits because last layer in network is not an activation function!
                        loss = criterion(outputs, labels.float())

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    outputs = torch.sigmoid(outputs)
                    if phase == 'val':
                        valid_preds[i * batch_size: (i+1) * batch_size] = outputs.cpu().numpy()
                    confmat = bcm(outputs, labels)
                    TN += confmat[0][0]
                    FP += confmat[0][1]
                    FN += confmat[1][0]
                    TP += confmat[1][1]
                    
                    avg_loss += loss.item() / len(dataloaders[phase])
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += metric(outputs, labels) 
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / len(dataloaders[phase])
                epoch_acc = running_corrects.double() / len(dataloaders[phase])

                precision = TP/(TP+FP)
                recall = TP/(TP+FN)
                F1 = (2*precision*recall)/(precision+recall)
                print(f'{phase} Loss: {epoch_loss:.4f} {phase} avg loss: {avg_loss:.4f} Acc: {epoch_acc:.4f} Precision: {precision:.4f} Recall: {recall:.4f} F1: {F1:.4f} TN: {TN} FP: {FP} FN: {FN} TP: {TP}')
                
                if phase == 'val':
                    score, weight = calculate_per_class_lwlrap(y_val, valid_preds)
                    lwlrap = (score * weight).sum()
                    print(f"val_lwlrap: {lwlrap:.6f}")

                # deep copy the model
                if phase == 'val' and lwlrap > best_lwlrap:
                    best_lwlrap = lwlrap
                    best_epoch = epoch
                    torch.save(model.state_dict(), best_model_params_path)
                

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best lwlrap: {best_lwlrap:4f} in epoch: {best_epoch}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model

def _one_sample_positive_class_precisions(scores, truth):
    """Calculate precisions for each true class for a single sample.

    Args:
      scores: np.array of (num_classes,) giving the individual classifier scores.
      truth: np.array of (num_classes,) bools indicating which classes are true.

    Returns:
      pos_class_indices: np.array of indices of the true classes for this sample.
      pos_class_precisions: np.array of precisions corresponding to each of those
        classes.
    """
    num_classes = scores.shape[0]
    pos_class_indices = np.flatnonzero(truth > 0)
    # Only calculate precisions if there are some true classes.
    if not len(pos_class_indices):
        return pos_class_indices, np.zeros(0)
    # Retrieval list of classes for this sample.
    retrieved_classes = np.argsort(scores)[::-1]
    # class_rankings[top_scoring_class_index] == 0 etc.
    class_rankings = np.zeros(num_classes, dtype=np.int32)
    class_rankings[retrieved_classes] = range(num_classes)
    # Which of these is a true label?
    retrieved_class_true = np.zeros(num_classes, dtype=np.bool_)
    retrieved_class_true[class_rankings[pos_class_indices]] = True
    # Num hits for every truncated retrieval list.
    retrieved_cumulative_hits = np.cumsum(retrieved_class_true)
    # Precision of retrieval list truncated at each hit, in order of pos_labels.
    precision_at_hits = (
            retrieved_cumulative_hits[class_rankings[pos_class_indices]] /
            (1 + class_rankings[pos_class_indices].astype(np.float64)))
    return pos_class_indices, precision_at_hits

def calculate_per_class_lwlrap(truth, scores):
    """Calculate label-weighted label-ranking average precision.

    Arguments:
      truth: np.array of (num_samples, num_classes) giving boolean ground-truth
        of presence of that class in that sample.
      scores: np.array of (num_samples, num_classes) giving the classifier-under-
        test's real-valued score for each class for each sample.

    Returns:
      per_class_lwlrap: np.array of (num_classes,) giving the lwlrap for each
        class.
      weight_per_class: np.array of (num_classes,) giving the prior of each
        class within the truth labels.  Then the overall unbalanced lwlrap is
        simply np.sum(per_class_lwlrap * weight_per_class)
    """
    labels = []
    for label in truth:
        one_hot_vector = np.zeros(len(classes), dtype=int)
        one_hot_vector[label] = 1
        labels.append(one_hot_vector)

    truth = np.array(labels)
    scores = np.array(scores)
    assert truth.shape == scores.shape
    num_samples, num_classes = scores.shape
    # Space to store a distinct precision value for each class on each sample.
    # Only the classes that are true for each sample will be filled in.
    precisions_for_samples_by_classes = np.zeros((num_samples, num_classes))
    for sample_num in range(num_samples):
        pos_class_indices, precision_at_hits = (
            _one_sample_positive_class_precisions(scores[sample_num, :],
                                                  truth[sample_num, :]))
        precisions_for_samples_by_classes[sample_num, pos_class_indices] = (
            precision_at_hits)
    labels_per_class = np.sum(truth > 0, axis=0)
    weight_per_class = labels_per_class / float(np.sum(labels_per_class))
    # Form average of each column, i.e. all the precisions assigned to labels in
    # a particular class.
    per_class_lwlrap = (np.sum(precisions_for_samples_by_classes, axis=0) /
                        np.maximum(1, labels_per_class))
    # overall_lwlrap = simple average of all the actual per-class, per-sample precisions
    #                = np.sum(precisions_for_samples_by_classes) / np.sum(precisions_for_samples_by_classes > 0)
    #           also = weighted mean of per-class lwlraps, weighted by class label prior across samples
    #                = np.sum(per_class_lwlrap * weight_per_class)
    return per_class_lwlrap, weight_per_class

if __name__ == "__main__":
    main()