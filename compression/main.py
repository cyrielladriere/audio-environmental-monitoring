# https://pytorch.org/blog/torchvision-mobilenet-v3-implementation/
import torch
import torchvision
import os
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.quantization import mobilenet_v3_large, MobileNet_V3_Large_QuantizedWeights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_model_size(mdl):
    torch.save(mdl.state_dict(), "tmp.pt")
    print("%.2f MB" %(os.path.getsize("tmp.pt")/1e6))
    os.remove('tmp.pt')

def main():
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1).to(device=device)
    model_quantized = mobilenet_v3_large(weights=MobileNet_V3_Large_QuantizedWeights.DEFAULT, quantize=True).to(device=device)
    print_model_size(model)
    print_model_size(model_quantized)

if __name__ == "__main__":
    main()