import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEResNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, cardinality=32, base_width=4, reduction=16):
        super(SEResNeXtBlock, self).__init__()
        
        width = int(out_channels * (base_width / 64.)) * cardinality
        
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width, track_running_stats=False)
        
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(width, track_running_stats=False)
        
        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels, track_running_stats=False)
        
        self.se = SEBlock(out_channels, reduction)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels, track_running_stats=False)
            )

    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        
        if self.shortcut is not None:
            residual = self.shortcut(x)
        
        out += residual
        out = F.relu(out)
        
        return out
    
class EncoderConv(nn.Module):
    def __init__(self, config):
        super(EncoderConv, self).__init__()
        self.latents = config['latents']
        self.c = config['initial_filters']

        number_of_layers = 6

        self.hw = config['inputsize'] // (2**number_of_layers)
        self.filters = [self.c * 2**i for i in range(number_of_layers)]
        
        self.maxpool = nn.MaxPool2d(2)

        if config['raster_as_input']:
            input_channels = len(config['organs'])
        else:
            input_channels = 1

        # Create downsampling layers dynamically
        self.dconv_down = nn.ModuleList()
        for i in range(len(self.filters)):
            in_channels = input_channels if i == 0 else self.filters[i-1]
            out_channels = self.filters[i]
            self.dconv_down.append(SEResNeXtBlock(in_channels, out_channels))
        
        # Final convolutional layer
        self.dconv_final = SEResNeXtBlock(self.filters[-1], self.filters[-1])


    def forward(self, x):
        conv_outputs = []
        
        for i, dconv in enumerate(self.dconv_down):
            x = dconv(x)
            conv_outputs.append(x)
            x = self.maxpool(x)
        
        x = self.dconv_final(x)
        
        
        return x, list(reversed(conv_outputs))


class DecoderConv(nn.Module):
    def __init__(self, config):
        super(DecoderConv, self).__init__()
        number_of_layers = 6
        
        self.hw = config['inputsize'] // (2**number_of_layers)

        self.filters = [self.c * 2**i for i in range(number_of_layers)]
        self.filters = self.filters + [self.filters[-1]]

        # Final output channels
        self.out_channels = len(config['organs'])
        
        # Create upsampling layers dynamically
        self.upconv = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()
        
        # Create transposed conv blocks (upsampling)
        for i, j in enumerate(range(len(self.filters)-1, 0, -1)):
            # Upsample from current decoder layer to next decoder layer
            self.upconv.append(nn.ConvTranspose2d(
                self.filters[j], self.filters[j-1], 
                kernel_size=2, stride=2, padding=0
            ))
            
            # Calculate combined channels after concatenation
            combined_channels = self.filters[j-1] * 2
            
            # Create the SEResNeXtBlock with the correct input channel count
            self.conv_blocks.append(SEResNeXtBlock(combined_channels, self.filters[j-1]))
        
        # Final convolution layer
        self.final_conv = nn.Conv2d(self.filters[0], self.out_channels, kernel_size=1)
        
    def forward(self, conv_outputs):
        # Upsampling path
        for i in range(1, len(self.filters)-1):
            # Upsample
            x = self.upconv[i](x)
            
            # Skip connections from encoder
            encoder_features = conv_outputs[i]
            
            # Concatenate along channel dimension
            x = torch.cat([x, encoder_features], dim=1)
            
            # Apply convolutional block directly with correct channel count
            x = self.conv_blocks[i](x)
            
            # Store features at each level for the graph decoder
            decoder_features.append(x)
        
        # Final convolution to get segmentation output
        segmentation = torch.sigmoid(self.final_conv(x))
        
        return segmentation