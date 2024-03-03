# Fusing: https://github.com/Sanjana7395/static_quantization/blob/master/quantization%20pytorch.ipynb
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from compression.training import train_model
from compression.models.PANN_pretrained import MobileNetV2
import torch
from torch import nn, optim
from compression.evaluation import print_model_size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pann_qat_v1(TENSORBOARD, model_pann, n_classes, dataloaders, n_epochs, data, threshold, batch_size):
    # Tensorboard
    today = datetime.now()
    date = today.strftime('%b%d_%y-%H-%M')
    model_dir = f"compression/runs/PANN_QAT/{date}"
    if TENSORBOARD: writer = SummaryWriter(model_dir)

    model = MobileNetV2(44100, 1024, 320, 64, 50, 14000, 527, quantize=True).to(device)
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

    model.cuda()
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')
    torch.ao.quantization.prepare_qat(model, inplace=True)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    exp_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
    if TENSORBOARD:
        model = train_model(model, dataloaders, optimizer, exp_lr_scheduler, n_epochs, data, threshold, batch_size, True, TENSORBOARD, writer)
        writer.flush()
        writer.close()

        model.to("cpu") # Needed for quatization convert
        model_qat = torch.quantization.convert(model.eval(), inplace=False)

        torch.save(model_qat.state_dict(), f"{model_dir}/model_pann_qat.pt")
    else:
        model = train_model(model, dataloaders, optimizer, exp_lr_scheduler, n_epochs, data, threshold, batch_size, True, TENSORBOARD)
        model.to("cpu") # Needed for quatization convert
        model_qat = torch.quantization.convert(model.eval(), inplace=False)
    print_model_size(model_qat)
    return model_qat

def pann_qat_v2(TENSORBOARD, model_pann_trained, dataloaders, n_epochs, data, threshold, batch_size):
    # Tensorboard
    today = datetime.now()
    date = today.strftime('%b%d_%y-%H-%M')
    model_dir = f"compression/runs/PANN_QAT_v2/{date}"
    if TENSORBOARD: writer = SummaryWriter(model_dir)

    model = MobileNetV2(44100, 1024, 320, 64, 50, 14000, 80, quantize=True, post_training=True).to(device)
    pretrained_weights = torch.load(model_pann_trained)
    model.load_state_dict(pretrained_weights)

    print_model_size(model)

    model.cuda()
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')
    torch.ao.quantization.prepare_qat(model, inplace=True)
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    exp_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)
    if TENSORBOARD:
        model = train_model(model, dataloaders, optimizer, exp_lr_scheduler, n_epochs, data, threshold, batch_size, True, TENSORBOARD, writer)
        writer.flush()
        writer.close()

        model.to("cpu") # Needed for quatization convert
        model_qat = torch.quantization.convert(model.eval(), inplace=False)

        torch.save(model_qat.state_dict(), f"{model_dir}/model_pann_qat_v2.pt")
    else:
        model = train_model(model, dataloaders, optimizer, exp_lr_scheduler, n_epochs, data, threshold, batch_size, True, TENSORBOARD)
        model.to("cpu") # Needed for quatization convert
        model_qat = torch.quantization.convert(model.eval(), inplace=False)
    print_model_size(model_qat)
    return model_qat


def pann_sq(model_pann_trained, dataloaders):
    model = MobileNetV2(44100, 1024, 320, 64, 50, 14000, 80, quantize=True, post_training=True)
    pretrained_weights = torch.load(model_pann_trained)
    model.load_state_dict(pretrained_weights)

    print_model_size(model) 

    model.qconfig = torch.ao.quantization.get_default_qconfig('x86')
    torch.backends.quantized.engine = 'x86'
    model = torch.quantization.prepare(model)

    # calibrate the prepared model to determine quantization parameters for activations
    # in a real world setting, the calibration would be done with a representative dataset (https://pytorch.org/docs/stable/quantization.html)
    for data in dataloaders["train"]:
        inputs, labels = data  # Shape inputs: [batch_size, channels, height, width]
        model(inputs)
    for data in dataloaders["val"]:
        inputs, labels = data  # Shape inputs: [batch_size, channels, height, width]
        model(inputs)

    model_static_quantized = torch.quantization.convert(model, inplace=False)
                
    torch.save(model_static_quantized.state_dict(), f"resources/model_pann_sq.pt")
    print_model_size(model_static_quantized)
    return model_static_quantized
    