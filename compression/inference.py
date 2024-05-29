import pandas as pd
from sklearn.model_selection import train_test_split
from compression.evaluation import calculate_per_class_lwlrap, print_model_size
from compression.models.PANN_pruned import Cnn14_pruned, MobileNetV2_pruned
from compression.preprocessing import convert_dataset, load_pkl, get_labels, convert_labels
from torchvision import transforms, datasets
from compression.main import TrainDataset
from thop import profile
from torch.utils.data import DataLoader, Dataset
import torch
import os
import numpy as np
import time
from compression.models.PANN_pretrained import Cnn14, MobileNetV2
from torchmetrics.classification import BinaryAccuracy, BinaryConfusionMatrix
device = "cpu"
# ------------- Testing Env
MODEL_PANN = False
PANN_QAT = False
PANN_QAT_V2 = False      
PANN_SQ = True         
OPNORM_PRUNING = False; P=0.91
L1_PRUNING = False
# ------------- Variables
training_audio_labels = "data/audio/train_curated.csv"
training_audio_data = "data/audio/train_curated"
audio_data = "data/train_curated"
model_pann = "resources/cnn_14/model_pann_cnn_14.pt"
model_pann_qat = "resources/cnn_14/model_pann_qat.pt"
model_pann_qat_v2 = "resources/cnn_14/model_pann_qat_v2.pt"
model_pann_sq = "resources/cnn_14/model_pann_cnn_14_sq.pt"
model_pann_opnorm_pruning = f"resources/cnn_14/model_opnorm_pruning_{P}_FT.pt"
model_pann_l1_pruning = f"resources/cnn_14/model_L1_norm_pruning_{P}_FT.pt"
# ------------- Hyperparameters
image_size = (256, 128)
batch_size = 64
threshold = 0.5

def main():
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    if not os.path.isfile(audio_data):
        convert_dataset(pd.read_csv(training_audio_labels), training_audio_data, audio_data) 
    data = load_pkl(audio_data)

    global labels_global
    labels_global = get_labels(data.keys())
    labels_global = convert_labels(labels_global)
    
    data_ = train_test_split(list(data.values()), labels_global, test_size=0.2, random_state=10)    # Random state makes sure we get same splits everytime (on same dataset)
    x_trn, x_val, y_trn, y_val = data_

    labels_global = y_val

    # Only validation set
    dataset = TrainDataset(x_val, y_val, transform)

    global nr_instances
    nr_instances = len(x_val)

    dataloader = DataLoader(dataset, batch_size=batch_size)

    if(MODEL_PANN):
        model = Cnn14(44100, 512, 320, 64, 50, 14000, 80, post_training=True).to(device)
        pretrained_weights = torch.load(model_pann)
        model.load_state_dict(pretrained_weights)
        # model.cuda()
        model.eval()
        predict(model, dataloader)
    elif(PANN_QAT):
        model = Cnn14(44100, 512, 320, 64, 50, 14000, 80, post_training=True, quantize=True)
        model.to("cpu")
        pretrained_weights = torch.load(model_pann_qat)

        model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')
        torch.ao.quantization.prepare_qat(model, inplace=True)
        model = torch.quantization.convert(model.eval(), inplace=True)

        model.load_state_dict(pretrained_weights)
        model.eval()
        predict(model, dataloader)
    elif(PANN_QAT_V2):
        model = Cnn14(44100, 512, 320, 64, 50, 14000, 80, post_training=True, quantize=True)
        model.to("cpu")
        pretrained_weights = torch.load(model_pann_qat_v2)

        model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')
        torch.ao.quantization.prepare_qat(model, inplace=True)
        model = torch.quantization.convert(model.eval(), inplace=True)

        model.load_state_dict(pretrained_weights)
        model.eval()
        predict(model, dataloader)

    elif(PANN_SQ):
        model = Cnn14(44100, 512, 320, 64, 50, 14000, 80, quantize=True, post_training=True)
        pretrained_weights = torch.load(model_pann_sq)

        model.qconfig = torch.ao.quantization.get_default_qconfig('x86')
        torch.backends.quantized.engine = 'x86'
        model = torch.quantization.prepare(model, inplace=True)
        model = torch.quantization.convert(model, inplace=True)

        model.load_state_dict(pretrained_weights)
        model.eval()
        predict(model, dataloader)
    elif(OPNORM_PRUNING):
        model = Cnn14_pruned(P, 44100, 512, 320, 64, 50, 14000, 80)
        pretrained_weights = torch.load(model_pann_opnorm_pruning)

        model.load_state_dict(pretrained_weights)
        model.eval()
        predict(model, dataloader)
    
    elif(L1_PRUNING):
        model = Cnn14_pruned(P, 44100, 512, 320, 64, 50, 14000, 80)
        pretrained_weights = torch.load(model_pann_l1_pruning)

        model.load_state_dict(pretrained_weights)
        model.eval()
        predict(model, dataloader)

    print_model_size(model, macs=True)

        
def predict(model, dataloader):
    start = time.time()
    preds = np.zeros((nr_instances, 80))

    TN = 0 
    FP = 0 
    FN = 0 
    TP = 0 

    avg_time = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            start_avg = time.time()
            inputs, labels = data
            inputs = inputs.to(device) # Shape: [batch_size, channels, height, width]
            labels = labels.to(device) # Shape: [batch_size, num_classes]

            bcm = BinaryConfusionMatrix(threshold=threshold).to(device)
            
            
            if MODEL_PANN or PANN_QAT or PANN_QAT_V2 or OPNORM_PRUNING or PANN_SQ or L1_PRUNING:
                # Outputs: {"clipwise_output": [batch_size, num_classes], "Embedding": }
                outputs = model(inputs)["clipwise_output"]
            else:
                outputs = model(inputs)
            
            outputs = torch.sigmoid(outputs)

            preds[i * batch_size: (i+1) * batch_size] = outputs.cpu().numpy()

            confmat = bcm(outputs, labels)
            TN += confmat[0][0]
            FP += confmat[0][1]
            FN += confmat[1][0]
            TP += confmat[1][1]
            
            end_avg = time.time()
            avg_time += end_avg-start_avg

        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        F1 = (2*precision*recall)/(precision+recall)

        score, weight = calculate_per_class_lwlrap(labels_global, preds)
        lwlrap = (score * weight).sum()
        end = time.time()
        # for x in score:
        #     print(f"{x},", end="")
        print("\n-----------------------------------")
        print(f'lwlrap: {lwlrap:.4f} Precision: {precision:.4f} Recall: {recall:.4f} F1: {F1:.4f} TN: {TN} FP: {FP} FN: {FN} TP: {TP}')
        print(f"Total time: {end-start:.2f}s, average inference time for 1batch: {(avg_time/nr_instances)*1000:.4f}ms")

if __name__ == "__main__":
    main()
