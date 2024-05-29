import torch
from torch import nn
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
import torch.nn.functional as F
from torch.nn.quantized import FloatFunctional
from torch.ao.quantization import DeQuantStub, QuantStub
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, P, quantize=False):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.quantize=quantize
        if(self.quantize):
            self.ff = FloatFunctional()

        hidden_dim = round(inp * expand_ratio)
        filters = int(hidden_dim*(1-P))
        self.use_res_connect = self.stride == 1 and int(inp*(1-P)) == oup

        if expand_ratio == 1:
            _layers = [
                nn.Conv2d(filters, filters, 3, 1, 1, groups=filters, bias=False), 
                nn.AvgPool2d(stride), 
                nn.BatchNorm2d(filters), 
                nn.ReLU6(inplace=True), 
                nn.Conv2d(filters, oup, 1, 1, 0, bias=False), 
                nn.BatchNorm2d(oup)
                ]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[2])
            init_layer(_layers[4])
            init_bn(_layers[5])
            self.conv = _layers
        else:
            _layers = [
                nn.Conv2d(int(inp*(1-P)), filters, 1, 1, 0, bias=False), 
                nn.BatchNorm2d(filters), 
                nn.ReLU6(inplace=True), 
                nn.Conv2d(filters, filters, 3, 1, 1, groups=filters, bias=False), 
                nn.AvgPool2d(stride), 
                nn.BatchNorm2d(filters), 
                nn.ReLU6(inplace=True), 
                nn.Conv2d(filters, oup, 1, 1, 0, bias=False), 
                nn.BatchNorm2d(oup)
                ]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[1])
            init_layer(_layers[3])
            init_bn(_layers[5])
            init_layer(_layers[7])
            init_bn(_layers[8])
            self.conv = _layers

    def forward(self, x):
        if self.use_res_connect:
            if self.quantize:
                return self.ff.add(x, self.conv(x))
            else:
                return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2_pruned(nn.Module):
    def __init__(self, P, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num, quantize=False):
        
        super(MobileNetV2_pruned, self).__init__()
        
        self.P = P
        self.quantize = quantize

        if self.quantize:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()
            self.ff = FloatFunctional()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

 
        self.bn0 = nn.BatchNorm2d(128) 
 
        width_mult=1.
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 2],
            [6, 160, 3, 1],
            [6, 320, 1, 1],
        ]

        def conv_bn(inp, oup, stride):
            _layers = [
                nn.Conv2d(inp, int(oup*(1-P)), 3, 1, 1, bias=False), 
                nn.AvgPool2d(stride), 
                nn.BatchNorm2d(int(oup*(1-P))), 
                nn.ReLU6(inplace=True)
                ]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[2])
            return _layers


        def conv_1x1_bn(inp, oup):
            _layers = nn.Sequential(
                nn.Conv2d(int(inp*(1-P)), int(oup*(1-P)), 1, 1, 0, bias=False),
                nn.BatchNorm2d(int(oup*(1-P))),
                nn.ReLU6(inplace=True)
            )
            init_layer(_layers[0])
            init_bn(_layers[1])
            return _layers

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(1, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, int(output_channel*(1-self.P)), s, expand_ratio=t, P=self.P, quantize=self.quantize))
                else:
                    self.features.append(block(input_channel, int(output_channel*(1-self.P)), 1, expand_ratio=t, P=self.P, quantize=self.quantize))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        self.fc1 = nn.Linear(in_features=int(1280*(1-P)), out_features=256, bias=True)    
        self.fc_audioset = nn.Linear(256, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)
 
    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        # x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        # x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        if self.quantize:
            input = self.quant(input)

        x = input.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.spec_augmenter(x)

        # # Mixup on spectrogram
        # if self.training and mixup_lambda is not None:
        #     x = do_mixup(x, mixup_lambda)
        
        x = self.features(x)
        
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        if self.quantize:
            x = self.ff.add(x1, x2)
        else:
            x = x1 + x2
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)

        clipwise_output = self.fc_audioset(x)

        if self.quantize:
            clipwise_output = self.dequant(clipwise_output)
        
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x
    
class Cnn14_pruned(nn.Module):
    def __init__(self, P, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num, quantize=False):
        self.quantize = quantize
        self.P = P

        super(Cnn14_pruned, self).__init__()
        if quantize:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()
            self.ff = FloatFunctional()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(128)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=int(64*(1-P)))
        self.conv_block2 = ConvBlock(in_channels=int(64*(1-P)), out_channels=int(128*(1-P)))
        self.conv_block3 = ConvBlock(in_channels=int(128*(1-P)), out_channels=int(256*(1-P)))
        self.conv_block4 = ConvBlock(in_channels=int(256*(1-P)), out_channels=int(512*(1-P)))
        self.conv_block5 = ConvBlock(in_channels=int(512*(1-P)), out_channels=int(1024*(1-P)))
        self.conv_block6 = ConvBlock(in_channels=int(1024*(1-P)), out_channels=int(2048*(1-P)))


        self.fc1 = nn.Linear(in_features=int(2048*(1-P)), out_features=256, bias=True)    # out_features tested: 1024(pretty bad), 512(ok), 128(okok)
        self.fc_audioset = nn.Linear(256, classes_num, bias=True)

        
        
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)
 
    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        # x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        # x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        if self.quantize:
            input = self.quant(input)

        x = input.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        # if self.training and mixup_lambda is not None:
        #     x = do_mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        if self.quantize:
            x = self.ff.add(x1, x2)
        else:
            x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = self.fc_audioset(x)

        if self.quantize:
            clipwise_output = self.dequant(clipwise_output)
        
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict
    
def loss_func(output, target):
    loss = - torch.mean(target * output)
    return loss