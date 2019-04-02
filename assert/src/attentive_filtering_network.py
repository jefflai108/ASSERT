from __future__ import print_function
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import SpoofResNet257_400, SpoofResNet257_500, SpoofSmallResNet257_400, SpoofSmallResNet256_400

# PyTorch implementation of Attentive Filtering Network

class AFNBasicBlock(nn.Module):
    ''' basic Conv2D Block for AFN '''
    def __init__(self, in_planes, out_planes, dilation=(1,1)):

        super(AFNBasicBlock, self).__init__()
        
        self.cnn = nn.Conv2d(in_planes, out_planes, kernel_size=(3,3), padding=(1,1), 
            dilation=dilation, bias=False)
        self.bn  = nn.BatchNorm2d(out_planes)
        self.re  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.re(self.bn(self.cnn(x)))


class AFNUpsamplingBlock(nn.Module):
    ''' basic upsampling Block for AFN '''
    def __init__(self, in_planes, out_planes, kernel, stride, uptype='Conv', size=None):

        super(AFNUpsamplingBlock, self).__init__()
        
        if uptype == 'Conv':
            self.up = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel, stride=stride, bias=False)
        elif uptype == 'Bilinear':
            self.up = nn.UpsamplingBilinear2d(size=size)
        self.bn  = nn.BatchNorm2d(out_planes)
        self.re  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.re(self.bn(self.up(x)))


class SpoofSmallAFNet256_400(nn.Module):
    ''' small attentive filtering network (less GPU mem usage) for feature map 257 by 400 '''
    def __init__(self, num_classes, upsampling_type='Bilinear', atten_activation='softmaxF', resnet_blocks=1, input_size=(1,256,400)):

        super(SpoofSmallAFNet256_400, self).__init__()
        
        ## attentive-filtering: bottom-up
        self.pre = nn.Sequential( # channel expansion 
            AFNBasicBlock(1, 4),
            AFNBasicBlock(4, 8)
        )

        self.down1 = nn.MaxPool2d(kernel_size=(2,2))
        self.att1  = AFNBasicBlock(8, 8, dilation=(2,2))
        self.skip1 = AFNBasicBlock(8, 8)

        self.down2 = nn.MaxPool2d(kernel_size=(2,2))
        self.att2  = AFNBasicBlock(8, 8, dilation=(4,4))
        self.skip2 = AFNBasicBlock(8, 8)

        self.down3 = nn.MaxPool2d(kernel_size=(2,2))
        self.att3  = AFNBasicBlock(8, 8, dilation=(4,4))
        self.skip3 = AFNBasicBlock(8, 8)

        self.down4 = nn.MaxPool2d(kernel_size=(2,2))
        self.att4  = nn.Sequential(
            AFNBasicBlock(8, 8, dilation=(4,8)),
            AFNBasicBlock(8, 8)
        )

        ## attentive-filtering: top-down 
        self.up5   = AFNUpsamplingBlock(8, 8, kernel=(2,5), stride=(5,7), uptype=upsampling_type, size=(22,40))
        self.att5  = AFNBasicBlock(8, 8)
        
        self.up6   = AFNUpsamplingBlock(8, 8, kernel=(15,15), stride=(2,2), uptype=upsampling_type, size=(57,93))
        self.att6  = AFNBasicBlock(8, 8)

        self.up7   = AFNUpsamplingBlock(8, 8, kernel=(14,14), stride=(2,2), uptype=upsampling_type, size=(126,198))
        self.att7  = AFNBasicBlock(8, 8)
        
        self.up8  = AFNUpsamplingBlock(8, 8, kernel=(6,6), stride=(2,2), uptype=upsampling_type, size=(256,400))
        self.post  = nn.Sequential( # channel compression
            AFNBasicBlock(8, 4),
            nn.Conv2d(4, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1) # omit the ReLU layer 
        )
        
        if atten_activation == 'softmaxF':
            self.soft = nn.Softmax(dim=2)
        if atten_activation == 'softmaxT':
            self.soft = nn.Softmax(dim=3)
        if atten_activation == 'sigmoid':
            self.soft  = nn.Sigmoid()
        
        self.resnet = SpoofSmallResNet256_400(num_classes, resnet_blocks)

        ## Weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d or nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d or nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
 
    def forward(self, x):
        residual = x
        ## attention block: bottom-up
        x = self.att1(self.down1(self.pre(x)))
        #print(x.size())
        skip7 = self.skip1(x)
        x = self.att2(self.down2(x))
        #print(x.size())
        skip6 = self.skip2(x)
        x = self.att3(self.down3(x))
        #print(x.size())
        skip5 = self.skip3(x)
        x = self.att4(self.down4(x))
        #print(x.size())
        ## attention block: top-down 
        #print((self.up5(x)).size())
        x = self.att5(skip5 + self.up5(x))
        #print((self.up6(x)).size())
        x = self.att6(skip6 + self.up6(x))
        #print((self.up7(x)).size())
        x = self.att7(skip7 + self.up7(x))
        #print((self.up8(x)).size())
        x = self.post(self.up8(x))
        x = (1 + self.soft(x)) * residual 

        return self.resnet(x)


class SpoofSmallAFNet257_400(nn.Module):
    ''' small attentive filtering network (less GPU mem usage) for feature map 257 by 400 '''
    def __init__(self, num_classes, upsampling_type='Bilinear', atten_activation='softmaxF', resnet_blocks=1, focal_loss=False, input_size=(1,257,400)):

        super(SpoofSmallAFNet257_400, self).__init__()
        
        ## attentive-filtering: bottom-up
        self.pre = nn.Sequential( # channel expansion 
            AFNBasicBlock(1, 4),
            AFNBasicBlock(4, 8)
        )

        self.down1 = nn.MaxPool2d(kernel_size=(2,2))
        self.att1  = AFNBasicBlock(8, 8, dilation=(2,2))
        self.skip1 = AFNBasicBlock(8, 8)

        self.down2 = nn.MaxPool2d(kernel_size=(2,2))
        self.att2  = AFNBasicBlock(8, 8, dilation=(4,4))
        self.skip2 = AFNBasicBlock(8, 8)

        self.down3 = nn.MaxPool2d(kernel_size=(2,2))
        self.att3  = AFNBasicBlock(8, 8, dilation=(4,4))
        self.skip3 = AFNBasicBlock(8, 8)

        self.down4 = nn.MaxPool2d(kernel_size=(2,2))
        self.att4  = nn.Sequential(
            AFNBasicBlock(8, 8, dilation=(4,8)),
            AFNBasicBlock(8, 8)
        )

        ## attentive-filtering: top-down 
        self.up5   = AFNUpsamplingBlock(8, 8, kernel=(2,5), stride=(5,7), uptype=upsampling_type, size=(22,40))
        self.att5  = AFNBasicBlock(8, 8)
        
        self.up6   = AFNUpsamplingBlock(8, 8, kernel=(15,15), stride=(2,2), uptype=upsampling_type, size=(57,93))
        self.att6  = AFNBasicBlock(8, 8)

        self.up7   = AFNUpsamplingBlock(8, 8, kernel=(14,14), stride=(2,2), uptype=upsampling_type, size=(126,198))
        self.att7  = AFNBasicBlock(8, 8)
        
        self.up8  = AFNUpsamplingBlock(8, 8, kernel=(7,6), stride=(2,2), uptype=upsampling_type, size=(257,400))
        self.post  = nn.Sequential( # channel compression
            AFNBasicBlock(8, 4),
            nn.Conv2d(4, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1) # omit the ReLU layer 
        )
        
        if atten_activation == 'softmaxF':
            self.soft = nn.Softmax(dim=2)
        if atten_activation == 'softmaxT':
            self.soft = nn.Softmax(dim=3)
        if atten_activation == 'sigmoid':
            self.soft  = nn.Sigmoid()
        
        self.resnet = SpoofSmallResNet257_400(num_classes=num_classes, resnet_blocks=resnet_blocks, 
                focal_loss=focal_loss)
        
        ## Weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d or nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d or nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
 
    def forward(self, x):
        residual = x
        ## attention block: bottom-up
        x = self.att1(self.down1(self.pre(x)))
        skip7 = self.skip1(x)
        x = self.att2(self.down2(x))
        skip6 = self.skip2(x)
        x = self.att3(self.down3(x))
        skip5 = self.skip3(x)
        x = self.att4(self.down4(x))
        ## attention block: top-down 
        #print((self.up5(x)).size())
        x = self.att5(skip5 + self.up5(x))
        #print((self.up6(x)).size())
        x = self.att6(skip6 + self.up6(x))
        #print((self.up7(x)).size())
        x = self.att7(skip7 + self.up7(x))
        #print((self.up8(x)).size())
        x = self.post(self.up8(x))
        x = (1 + self.soft(x)) * residual 

        return self.resnet(x)


class SpoofAFNet257_500(nn.Module):
    ''' primative attentive filtering network for 257 by 500 feature map '''
    def __init__(self, num_classes, upsampling_type='Bilinear', atten_activation='softmaxF', resnet_blocks=1, input_size=(1,257,500)):

        super(SpoofAFNet257_500, self).__init__()
        
        ## attentive-filtering: bottom-up
        self.pre = nn.Sequential( # channel expansion 
            AFNBasicBlock(1, 4),
            AFNBasicBlock(4, 16)
        )

        self.down1 = nn.MaxPool2d(kernel_size=3, stride=1)
        self.att1  = AFNBasicBlock(16, 16, dilation=(8,16))
        self.skip1 = AFNBasicBlock(16, 16)

        self.down2 = nn.MaxPool2d(kernel_size=3, stride=1)
        self.att2  = AFNBasicBlock(16, 16, dilation=(8,16))
        self.skip2 = AFNBasicBlock(16, 16)

        self.down3 = nn.MaxPool2d(kernel_size=3, stride=1)
        self.att3  = AFNBasicBlock(16, 16, dilation=(16,32))
        self.skip3 = AFNBasicBlock(16, 16)

        self.down4 = nn.MaxPool2d(kernel_size=3, stride=1)
        self.att4  = AFNBasicBlock(16, 16, dilation=(32,64))
        self.skip4 = AFNBasicBlock(16, 16)

        self.down5 = nn.MaxPool2d(kernel_size=3, stride=1)
        self.att5  = nn.Sequential(
            AFNBasicBlock(16, 16, dilation=(64,114)), # change dilation from (64,64) to (64,114) 
            AFNBasicBlock(16, 16)
        )

        ## attentive-filtering: top-down 
        self.up6   = AFNUpsamplingBlock(16, 16, dilation=(64,114), uptype=upsampling_type, size=(129,244))
        self.att6  = AFNBasicBlock(16, 16)

        self.up7   = AFNUpsamplingBlock(16, 16, dilation=(32,64), uptype=upsampling_type, size=(193,372))
        self.att7  = AFNBasicBlock(16, 16)
        
        self.up8   = AFNUpsamplingBlock(16, 16, dilation=(16,32), uptype=upsampling_type, size=(225,436))
        self.att8  = AFNBasicBlock(16, 16)

        self.up9   = AFNUpsamplingBlock(16, 16, dilation=(8,16), uptype=upsampling_type, size=(241,468))
        self.att9  = AFNBasicBlock(16, 16)
        
        self.up10  = AFNUpsamplingBlock(16, 16, dilation=(8,16), uptype=upsampling_type, size=(257,500))
        self.post  = nn.Sequential( # channel compression
            AFNBasicBlock(16, 4),
            nn.Conv2d(4, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1) # omit the ReLU layer 
        )
 
        if atten_activation == 'softmaxF':
            self.soft = nn.Softmax(dim=2)
        if atten_activation == 'softmaxT':
            self.soft = nn.Softmax(dim=3)
        if atten_activation == 'sigmoid':
            self.soft  = nn.Sigmoid()
        
        self.resnet = SpoofResNet257_500(num_classes, resnet_blocks)

        ## Weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d or nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d or nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
 
    def forward(self, x):
        residual = x
        ## attention block: bottom-up
        x = self.att1(self.down1(self.pre(x)))
        ##print(x.size())
        skip9 = self.skip1(x)
        x = self.att2(self.down2(x))
        ##print(x.size())
        skip8 = self.skip2(x)
        x = self.att3(self.down3(x))
        ##print(x.size())
        skip7 = self.skip3(x)
        x = self.att4(self.down4(x))
        ##print(x.size())
        skip6 = self.skip4(x)
        x = self.att5(self.down5(x))
        ##print(x.size())
        ## attention block: top-down 
        x = self.post(self.up10(self.att9(skip9 + self.up9(self.att8(skip8 + self.up8(
            self.att7(skip7 + self.up7(self.att6(skip6 + self.up6(x))))))))))
        x = (1 + self.soft(x)) * residual 
        ##print(x.size())

        return self.resnet(x)


class SpoofAFNet257_400(nn.Module):
    ''' primative attentive filtering network for feature map 257 by 400 '''
    def __init__(self, num_classes, upsampling_type='Bilinear', atten_activation='softmaxF', resnet_blocks=1, input_size=(1,257,400)):

        super(SpoofAFNet257_400, self).__init__()
        
        ## attentive-filtering: bottom-up
        self.pre = nn.Sequential( # channel expansion 
            AFNBasicBlock(1, 4),
            AFNBasicBlock(4, 16)
        )

        self.down1 = nn.MaxPool2d(kernel_size=3, stride=1)
        self.att1  = AFNBasicBlock(16, 16, dilation=(8,16))
        self.skip1 = AFNBasicBlock(16, 16)

        self.down2 = nn.MaxPool2d(kernel_size=3, stride=1)
        self.att2  = AFNBasicBlock(16, 16, dilation=(8,16))
        self.skip2 = AFNBasicBlock(16, 16)

        self.down3 = nn.MaxPool2d(kernel_size=3, stride=1)
        self.att3  = AFNBasicBlock(16, 16, dilation=(16,32))
        self.skip3 = AFNBasicBlock(16, 16)

        self.down4 = nn.MaxPool2d(kernel_size=3, stride=1)
        self.att4  = AFNBasicBlock(16, 16, dilation=(32,64))
        self.skip4 = AFNBasicBlock(16, 16)

        self.down5 = nn.MaxPool2d(kernel_size=3, stride=1)
        self.att5  = nn.Sequential(
            AFNBasicBlock(16, 16, dilation=(64,64)),
            AFNBasicBlock(16, 16)
        )

        ## attentive-filtering: top-down 
        self.up6   = AFNUpsamplingBlock(16, 16, dilation=(64,64), uptype=upsampling_type, size=(129,144))
        self.att6  = AFNBasicBlock(16, 16)

        self.up7   = AFNUpsamplingBlock(16, 16, dilation=(32,64), uptype=upsampling_type, size=(193,272))
        self.att7  = AFNBasicBlock(16, 16)
        
        self.up8   = AFNUpsamplingBlock(16, 16, dilation=(16,32), uptype=upsampling_type, size=(225,336))
        self.att8  = AFNBasicBlock(16, 16)

        self.up9   = AFNUpsamplingBlock(16, 16, dilation=(8,16), uptype=upsampling_type, size=(241,368))
        self.att9  = AFNBasicBlock(16, 16)
        
        self.up10  = AFNUpsamplingBlock(16, 16, dilation=(8,16), uptype=upsampling_type, size=(257,400))
        self.post  = nn.Sequential( # channel compression
            AFNBasicBlock(16, 4),
            nn.Conv2d(4, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1) # omit the ReLU layer 
        )
        
        if atten_activation == 'softmaxF':
            self.soft = nn.Softmax(dim=2)
        if atten_activation == 'softmaxT':
            self.soft = nn.Softmax(dim=3)
        if atten_activation == 'sigmoid':
            self.soft  = nn.Sigmoid()
        
        self.resnet = SpoofResNet257_400(num_classes, resnet_blocks)

        ## Weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d or nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d or nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
 
    def forward(self, x):
        residual = x
        ## attention block: bottom-up
        x = self.att1(self.down1(self.pre(x)))
        ##print(x.size())
        skip9 = self.skip1(x)
        x = self.att2(self.down2(x))
        ##print(x.size())
        skip8 = self.skip2(x)
        x = self.att3(self.down3(x))
        ##print(x.size())
        skip7 = self.skip3(x)
        x = self.att4(self.down4(x))
        ##print(x.size())
        skip6 = self.skip4(x)
        x = self.att5(self.down5(x))
        ##print(x.size())
        ## attention block: top-down 
        x = self.post(self.up10(self.att9(skip9 + self.up9(self.att8(skip8 + self.up8(
            self.att7(skip7 + self.up7(self.att6(skip6 + self.up6(x))))))))))
        x = (1 + self.soft(x)) * residual 
        ##print(x.size())

        return self.resnet(x)

