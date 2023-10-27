import torch
from torch import nn
import torch.optim as optim
from torchvision.models.quantization import mobilenet_v3_large, MobileNet_V3_Large_QuantizedWeights
import time
import copy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = {"n01440764": 0, "n02102040": 217, "n02979186": 482, "n03000684": 491, "n03028079": 497, "n03394916": 566, "n03417042": 569, "n03425413": 571, "n03445777": 574, "n03888257": 701}

def qat(train_dataloader, val_dataloader):
    # You will need the number of filters in the `fc` for future use.
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_fe = mobilenet_v3_large(weights=MobileNet_V3_Large_QuantizedWeights.DEFAULT, quantize=True).to(device=device)
    num_ftrs = 96000 
    
    new_model = create_combined_model(model_fe, num_ftrs)
    new_model = new_model.to('cpu')

    criterion = nn.CrossEntropyLoss()

    # Note that we are only training the head.
    optimizer_ft = optim.SGD(new_model.parameters(), lr=0.01, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    new_model = train_model(new_model, train_dataloader, val_dataloader ,criterion, optimizer_ft, exp_lr_scheduler,
                            num_epochs=25, device=device)
    return new_model

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, num_epochs=25, device=device):
  """
  Support function for model training.

  Args:
    model: Model to be trained
    criterion: Optimization criterion (loss)
    optimizer: Optimizer to use for training
    scheduler: Instance of ``torch.optim.lr_scheduler``
    num_epochs: Number of epochs
    device: Device to run the training on. Must be 'cpu' or 'cuda'
  """
  since = time.time()

  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0

  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
      if phase == 'train':
        model.train()  # Set model to training mode
        data = train_dataloader
        dataset_size = len(train_dataloader) #..?
      else:
        model.eval()   # Set model to evaluate mode
        data = val_dataloader
        dataset_size = len(val_dataloader) #..?

      running_loss = 0.0
      running_corrects = 0

      # Iterate over data.
      for sample in data:
        inputs = sample['image'].permute(0, 3, 1, 2).float().to(device)         # change shape: [batch_size, height, width, channels] -> [batch_size, channels, height, width]
        labels = sample['label']
        labels = torch.tensor([classes[item] for item in labels]).to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
          outputs = model(inputs)
          print(outputs)
          _, preds = torch.max(outputs, 1)
          loss = criterion(outputs, labels)

          # backward + optimize only if in training phase
          
          if phase == 'train':
            loss.backward()
            optimizer.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
      if phase == 'train':
        scheduler.step()

      epoch_loss = running_loss / dataset_size
      epoch_acc = running_corrects.double() / dataset_size

      print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        phase, epoch_loss, epoch_acc))

      # deep copy the model
      if phase == 'val' and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())

    print()

  time_elapsed = time.time() - since
  print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
  print('Best val Acc: {:4f}'.format(best_acc))

  # load best model weights
  model.load_state_dict(best_model_wts)
  return model

def create_combined_model(model_fe, num_ftrs):
  # Step 1. Isolate the feature extractor.
  model_fe_features = nn.Sequential(
    model_fe.quant,  # Quantize the input
    model_fe.features,
    model_fe.dequant,  # Dequantize the output
  )

  # Step 2. Create a new "head"
  new_head = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(num_ftrs, 1000),
  )

  # Step 3. Combine, and don't forget the quant stubs.
  new_model = nn.Sequential(
    model_fe_features,
    nn.Flatten(1),
    new_head,
  )
  return new_model