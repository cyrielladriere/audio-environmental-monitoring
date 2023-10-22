# https://github.com/Sanjana7395/static_quantization/blob/master/quantization%20pytorch.ipynb
# https://pytorch.org/docs/master/quantization.html#quantization-aware-training
# https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html !
import torch
from torchvision.io import read_image
from torchvision.models.quantization import mobilenet_v3_large, MobileNet_V3_Large_QuantizedWeights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up warnings
import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.ao.quantization'
)

img = read_image("data/koala.jpg")

# Step 1: Initialize model with the best available weights
weights = MobileNet_V3_Large_QuantizedWeights.DEFAULT
model = mobilenet_v3_large(weights=weights, quantize=True).to(device=device)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)

# Step 4: Use the model and print the predicted category
prediction = model(batch).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score}%")