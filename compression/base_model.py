# https://harinramesh.medium.com/transfer-learning-in-pytorch-f7736598b1ed
# https://tiwari11-rst.medium.com/transfer-learning-part-6-2-implementing-mobilenet-in-pytorch-2d3f3851a15b
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
import torch
from torch import nn
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def base_model(model, train_loader, val_dataloader, classes, n_epochs=10):
    for param in model.parameters():
        param.requires_grad = False
    
    # print(model.classifier)
    num_ftrs = model.classifier[3].in_features  # model.classifier[0] = model.fc
    model.classifier[3] = nn.Linear(num_ftrs, 2)
    # model.classifier[3] = nn.Linear(num_ftrs, 2)   # (num_features, len(classes))

    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_model(model, train_loader, n_epochs, optimizer, criterion)

    validate_model(model, val_dataloader, classes)
    
    
    return model

def train_model(model, train_loader, n_epochs, optimizer, criterion):
    for epoch in range(n_epochs):
        running_loss = 0.0
       
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs) # Something goes wrong here with quantized model
            print("ok")
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        print(f"Epoch: {epoch+1}/{n_epochs}, running_loss: {running_loss}")
    print('Finished Training')

def validate_model(model, val_dataloader, classes):
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    with torch.no_grad():
        for i, data in enumerate(val_dataloader, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(4):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
    for i, cl in enumerate(classes):
        print('Val_accuracy of %5s : %2d %%' % (
            cl, 100 * class_correct[i] / class_total[i]))