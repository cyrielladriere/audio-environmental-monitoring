# https://github.com/Sanjana7395/static_quantization/blob/master/quantization%20pytorch.ipynb
# https://pytorch.org/docs/master/quantization.html#quantization-aware-training !
# https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html !
import torch
from torch import nn, Tensor
from torch.ao.quantization import DeQuantStub, QuantStub
from torchvision.models import MobileNetV3
from compression.evaluation import AverageMeter, accuracy, evaluate
import time
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

def QAT(data_loader_train, data_loader_val):
    # create a model instance
    model_fp32 = QuantizableMobileNetV3()

    # model must be set to eval for fusion to work
    model_fp32.eval()

    # attach a global qconfig, which contains information about what kind
    # of observers to attach. 
    # Use 'x86' for server inference and 'qnnpack'for mobile inference. 
    # Other quantization configurations such as selecting symmetric or asymmetric quantization and MinMax or L2Norm calibration techniques can be specified here.
    # Note: the old 'fbgemm' is still available but 'x86' is the recommended default for server inference.
    model_fp32.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')

    # fuse the activations to preceding layers, where applicable
    # this needs to be done manually depending on the model architecture
    model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32,
        [['conv', 'bn', 'relu']])

    # Prepare the model for QAT. This inserts observers and fake_quants in
    # the model needs to be set to train for QAT logic to work
    # the model that will observe weight and activation tensors during calibration.
    model_fp32_prepared = torch.ao.quantization.prepare_qat(model_fp32_fused.train())

    # run the training loop (not shown)
    training_loop(model_fp32_prepared, data_loader_train, data_loader_val)

    # Convert the observed model to a quantized model. This does several things:
    # quantizes the weights, computes and stores the scale and bias value to be
    # used with each activation tensor, fuses modules where appropriate,
    # and replaces key operators with quantized implementations.
    model_fp32_prepared.eval()
    model_int8 = torch.ao.quantization.convert(model_fp32_prepared)

    return model_int8

def training_loop(qat_model, n_epochs, data_loader_train, data_loader_val):
    batch_size = 64
    criterion = nn.CrossEntropyLoss
    optimizer = torch.optim.SGD(qat_model.parameters(), lr = 0.0001)
    # QAT takes time and one needs to train over a few epochs.
    # Train and check accuracy after each epoch
    for nepoch in range(n_epochs):
        train_one_epoch(qat_model, criterion, optimizer, data_loader_train, device, batch_size)
        if nepoch > 3:
            # Freeze quantizer parameters
            qat_model.apply(torch.ao.quantization.disable_observer)
        if nepoch > 2:
            # Freeze batch norm mean and variance estimates
            qat_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

        # Check the accuracy after each epoch
        quantized_model = torch.ao.quantization.convert(qat_model.eval(), inplace=False)
        quantized_model.eval()
        top1, top5 = evaluate(quantized_model, criterion, data_loader_val, neval_batches=batch_size)
        print('Epoch %d :Evaluation accuracy on %d images, %2.2f'%(nepoch, batch_size * batch_size, top1.avg))

def train_one_epoch(model, criterion, optimizer, data_loader, device, ntrain_batches):
    model.train()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    avgloss = AverageMeter('Loss', '1.5f')

    cnt = 0
    for image, target in data_loader:
        start_time = time.time()
        print('.', end = '')
        cnt += 1
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))
        avgloss.update(loss, image.size(0))
        if cnt >= ntrain_batches:
            print('Loss', avgloss.avg)

            print('Training: * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))
            return

    print('Full imagenet train set:  * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=top1, top5=top5))
    return

class QuantizableMobileNetV3(MobileNetV3):
    def __init__(self, num_classes=1000, inverted_residual_setting=None, last_channel=1280):
        """
        MobileNet V3 main class

        Args:
           Inherits args from floating point MobileNetV3
        """
        super().__init__(num_classes, inverted_residual_setting, last_channel)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self._forward_impl(x)
        x = self.dequant(x)
        return x