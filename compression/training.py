import time
from torchmetrics.classification import BinaryAccuracy, BinaryConfusionMatrix
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
import os
from tempfile import TemporaryDirectory
from compression.evaluation import calculate_per_class_lwlrap
import torch
from torch import nn
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = {'Bark': 0, 'Motorcycle': 1, 'Writing': 2, 'Female_speech_and_woman_speaking': 3, 'Tap': 4, 'Child_speech_and_kid_speaking': 5, 'Screaming': 6, 'Meow': 7, 'Scissors': 8, 'Fart': 9, 'Car_passing_by': 10, 'Harmonica': 11, 'Sink_(filling_or_washing)': 12, 'Burping_and_eructation': 13, 'Slam': 14, 'Drawer_open_or_close': 15, 'Cricket': 16, 'Hiss': 17, 'Frying_(food)': 18, 'Sneeze': 19, 'Chink_and_clink': 20, 'Fill_(with_liquid)': 21, 'Crowd': 22, 'Marimba_and_xylophone': 23, 'Sigh': 24, 'Accordion': 25, 'Electric_guitar': 26, 'Cupboard_open_or_close': 27, 'Bicycle_bell': 28, 'Waves_and_surf': 29, 'Stream': 30, 'Bus': 31, 'Toilet_flush': 32, 'Trickle_and_dribble': 33, 'Tick-tock': 34, 'Keys_jangling': 35, 'Acoustic_guitar': 36, 'Finger_snapping': 37, 'Cheering': 38, 'Race_car_and_auto_racing': 39, 'Bass_guitar': 40, 'Yell': 41, 'Water_tap_and_faucet': 42, 'Run': 43, 'Traffic_noise_and_roadway_noise': 44, 'Crackle': 45, 'Skateboard': 46, 'Glockenspiel': 47, 'Computer_keyboard': 48, 'Whispering': 49, 'Zipper_(clothing)': 50, 'Microwave_oven': 51, 'Bathtub_(filling_or_washing)': 52, 'Male_speech_and_man_speaking': 53, 'Gong': 54, 'Shatter': 55, 'Strum': 56, 'Bass_drum': 57, 'Dishes_and_pots_and_pans': 58, 'Accelerating_and_revving_and_vroom': 59, 'Male_singing': 60, 'Gurgling': 61, 'Walk_and_footsteps': 62, 'Printer': 63, 'Cutlery_and_silverware': 64, 'Chirp_and_tweet': 65, 'Clapping': 66, 'Hi-hat': 67, 'Raindrop': 68, 'Gasp': 69, 'Buzz': 70, 'Drip': 71, 'Chewing_and_mastication': 72, 'Squeak': 73, 'Female_singing': 74, 'Church_bell': 75, 'Mechanical_fan': 76, 'Purr': 77, 'Applause': 78, 'Knock': 79}

def train_model(model, dataloaders, optimizer, scheduler, num_epochs, data, threshold, batch_size, PANN, TENSORBOARD, writer=None):
    x_trn, x_val, y_trn, y_val = data
    
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_lwlrap = 0.0
        best_epoch_loss = float('inf')
        best_epoch = 0

        for epoch in range(1, num_epochs+1):
            start_epoch = time.time()
            print(f'Epoch {epoch}/{num_epochs}')
            print('-' * 10)

            valid_preds = np.zeros((len(x_val), len(classes)))
            train_preds = np.zeros((len(x_trn), len(classes)))

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
                    metric = BinaryAccuracy(threshold=threshold).to(device)
                    bcm = BinaryConfusionMatrix(threshold=threshold).to(device)
                    
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        if PANN:
                            # Outputs: {"clipwise_output": [batch_size, num_classes], "Embedding": }
                            outputs = model(inputs)["clipwise_output"]
                        else:
                            outputs = model(inputs)

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
                    else:
                        train_preds[i * batch_size: (i+1) * batch_size] = outputs.detach().cpu().numpy()
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
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Precision: {precision:.4f} Recall: {recall:.4f} F1: {F1:.4f} TN: {TN} FP: {FP} FN: {FN} TP: {TP}')
                if TENSORBOARD:
                    writer.add_scalar(f"lr", scheduler.get_lr()[0], epoch)
                    writer.add_scalar(f"loss/{phase}", epoch_loss, epoch)
                    writer.add_scalar(f"accuracy/{phase}", epoch_acc, epoch)
                    writer.add_scalar(f"precision/{phase}", precision, epoch)
                    writer.add_scalar(f"recall/{phase}", recall, epoch)
                    writer.add_scalar(f"f1/{phase}", F1, epoch)
                    writer.add_scalar(f"Confusion_Matrix/TN/{phase}", TN, epoch)
                    writer.add_scalar(f"Confusion_Matrix/FP/{phase}", FP, epoch)
                    writer.add_scalar(f"Confusion_Matrix/FN/{phase}", FN, epoch)
                    writer.add_scalar(f"Confusion_Matrix/TP/{phase}", TP, epoch)
                
                if phase == 'val':
                    score, weight = calculate_per_class_lwlrap(y_val, valid_preds)
                    lwlrap = (score * weight).sum()
                    end_epoch = time.time()
                    print(f"val_lwlrap: {lwlrap:.6f} epoch time: {end_epoch-start_epoch:.2f}s")
                    if TENSORBOARD:
                        writer.add_scalar(f"val_lwlrap", lwlrap, epoch)
                        writer.add_scalar(f"time", end_epoch-start_epoch, epoch)

                # deep copy the model
                if phase == 'val' and epoch_loss < best_epoch_loss:
                    best_lwlrap = lwlrap
                    best_epoch = epoch
                    best_epoch_loss = epoch_loss
                    torch.save(model.state_dict(), best_model_params_path)
            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val loss: {best_epoch_loss:.4f} lwlrap: {best_lwlrap:4f} in epoch: {best_epoch}')

        if(TENSORBOARD):
            trn_labels = []
            for label in y_trn:
                one_hot_vector = np.zeros(len(classes), dtype=int)
                one_hot_vector[label] = 1
                trn_labels.append(one_hot_vector)
            trn_truth = np.array(trn_labels)

            val_labels = []
            for label in y_val:
                one_hot_vector = np.zeros(len(classes), dtype=int)
                one_hot_vector[label] = 1
                val_labels.append(one_hot_vector)
            val_truth = np.array(val_labels)

            # Create PR-Curves in Tensorboard
            for cl, id in classes.items():
                tensorboard_truth = (trn_truth[:, id] == 1).astype(int)
                tensorboard_probs = train_preds[:, id]
                writer.add_pr_curve(f"{cl}/train", tensorboard_truth, tensorboard_probs)

                tensorboard_truth = (val_truth[:, id] == 1).astype(int)
                tensorboard_probs = valid_preds[:, id]
                writer.add_pr_curve(f"{cl}/val", tensorboard_truth, tensorboard_probs)
            
            # Create Confusion matrix in Tensorboard
            classes_list = list(classes)
            cf_matrix = multilabel_confusion_matrix(trn_truth, np.where(np.array(train_preds) > threshold, 1, 0))
            for i, cf in enumerate(cf_matrix):
                disp = ConfusionMatrixDisplay(cf)
                disp.plot()
                disp.ax_.set_title(f'class {classes_list[i]}')
                writer.add_figure(f"Train_ConfusionMatrices/{classes_list[i]}", disp.figure_, epoch)
            
            cf_matrix = multilabel_confusion_matrix(val_truth, np.where(np.array(valid_preds) > threshold, 1, 0))
            for i, cf in enumerate(cf_matrix):
                disp = ConfusionMatrixDisplay(cf)
                disp.plot()
                disp.ax_.set_title(f'class {classes_list[i]}')
                writer.add_figure(f"Val_ConfusionMatrices/{classes_list[i]}", disp.figure_, epoch)

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model