from compression.preprocessing import load_pkl, get_labels, convert_labels
from torchvision import transforms, datasets
from compression.main import TrainDataset, calculate_per_class_lwlrap
from torch.utils.data import DataLoader, Dataset
import torch
from torch import nn
import numpy as np
import time
from compression.models.PANN_pretrained import MobileNetV2
from torchmetrics.classification import BinaryAccuracy, BinaryConfusionMatrix
device = "cpu"

# ------------- Testing Env
MODEL_PANN = False
PANN_QAT = True
PANN_QAT_V2 = False      
PANN_SQ = False         
PRUNING = False
MODEL_AT = False
# ------------- Variables
audio_data = "data/train_curated"
model_pann = "resources/model_pann.pt"
model_pann_qat = "resources/model_pann_qat.pt"
model_pann_qat_v2 = "resources/model_pann_qat_v2.pt"
model_pann_sq = "resources/model_pann_sq.pt"
# ------------- Hyperparameters
image_size = (256, 128)
batch_size = 64
threshold = 0.5

def main():
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    data = load_pkl(audio_data)
    global labels_global
    labels_global = get_labels(data.keys())
    labels_global = convert_labels(labels_global)

    dataset = TrainDataset(list(data.values()), labels_global, transform)
    global nr_instances
    nr_instances = len(list(data.values()))

    dataloader = DataLoader(dataset, batch_size=batch_size)

    if(MODEL_PANN):
        model = MobileNetV2(44100, 1024, 320, 64, 50, 14000, 80, post_training=True).to(device)
        pretrained_weights = torch.load(model_pann)
        model.load_state_dict(pretrained_weights)
        model.cuda()
        model.eval()
        predict(model, dataloader)
    elif(PANN_QAT):
        # model = torch.jit.load('model_scripted.pt')
        # model.eval()
        # predict(model, dataloader)
        model = MobileNetV2(44100, 1024, 320, 64, 50, 14000, 80, post_training=True, quantize=True)
        model.to("cpu")
        pretrained_weights = torch.load(model_pann_qat)
        model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')
        torch.ao.quantization.prepare_qat(model, inplace=True)
        model = torch.quantization.convert(model.eval(), inplace=True)
        # https://discuss.pytorch.org/t/error-in-running-quantised-model-runtimeerror-could-not-run-quantized-conv2d-new-with-arguments-from-the-cpu-backend/151718/3
        model.load_state_dict(pretrained_weights)
        model.eval()
        predict(model, dataloader)
        # # model_fp32.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')
        # # torch.ao.quantization.prepare_qat(model_fp32, inplace=True)
        # # model_fp32.to("cpu") # Needed for quatization convert
        # # model = torch.quantization.convert(model_fp32.eval(), inplace=False)
        # # pretrained_weights = torch.load(model_pann_qat)
    elif(PANN_QAT_V2):
        pretrained_weights = torch.load(model_pann_qat_v2)
    elif(PANN_SQ):
        pretrained_weights = torch.load(model_pann_sq)
        
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
            
            
            if MODEL_PANN or PANN_QAT or PANN_QAT_V2:
                # Outputs: {"clipwise_output": [batch_size, num_classes], "Embedding": }
                outputs = model(inputs)["clipwise_output"]
            elif MODEL_AT:
                outputs = model(inputs)[0]
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
        print(f'lwlrap: {lwlrap:.4f} Precision: {precision:.4f} Recall: {recall:.4f} F1: {F1:.4f} TN: {TN} FP: {FP} FN: {FN} TP: {TP}')
        print(f"Total time: {end-start:.2f}s, average inference time for 1batch: {(avg_time/nr_instances)*1000:.4f}ms")

if __name__ == "__main__":
    main()