from __future__ import print_function
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch implementation of Dilated Residual Network

def conv3x3(planes):
    ''' 3x3 convolution '''
    return nn.Conv2d(planes, planes, kernel_size=(3,3), padding=(1,1), bias=False)

class ResBasicBlock(nn.Module):
    ''' basic Conv2D Block for ResNet '''
    def __init__(self, planes):

        super(ResBasicBlock, self).__init__()
        
        self.bn1  = nn.BatchNorm2d(planes)
        self.re1  = nn.ReLU(inplace=True)
        self.cnn1 = conv3x3(planes)
        self.bn2  = nn.BatchNorm2d(planes)
        self.re2  = nn.ReLU(inplace=True)
        self.cnn2 = conv3x3(planes)

    def forward(self, x):
        residual = x
        x = self.cnn2(self.re2(self.bn2(self.cnn1(self.re1(self.bn1(x))))))
        x += residual 
        
        return x

class SpoofSmallResNet256_400(nn.Module):
    ''' small ResNet for 256 by 400 feature map (same NN as SpoofSmallResNet257_400) '''
    def __init__(self, num_classes, binary=False, resnet_blocks=1, input_size=(1,256,400)):

        super(SpoofSmallResNet256_400, self).__init__()
        
        self.binary = binary 

        self.expansion = nn.Conv2d(1, 8, kernel_size=(3,3), padding=(1,1))
        ## block 1
        layers = []
        for i in range(resnet_blocks):
            layers.append(ResBasicBlock(8))
        self.block1 = nn.Sequential(*layers)
        self.mp1    = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn1   = nn.Conv2d(8, 16, kernel_size=(3,3), dilation=(2,2))
        ## block 2
	layers = []
        for i in range(resnet_blocks):
            layers.append(ResBasicBlock(16))
        self.block2 = nn.Sequential(*layers)
        self.mp2    = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn2   = nn.Conv2d(16, 32, kernel_size=(3,3), dilation=(4,4))
        ## block 3
	layers = []
        for i in range(resnet_blocks):
            layers.append(ResBasicBlock(32))
        self.block3 = nn.Sequential(*layers)
        self.mp3    = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn3   = nn.Conv2d(32, 64, kernel_size=(3,3), dilation=(4,4))
        ## block 4
	layers = []
        for i in range(resnet_blocks):
            layers.append(ResBasicBlock(64))
        self.block4 = nn.Sequential(*layers)
        self.mp4    = nn.MaxPool2d(kernel_size=(1,2))
        self.cnn4   = nn.Conv2d(64, 64, kernel_size=(3,3), dilation=(8,8)) 

        self.flat_feats = 64*3*2

        self.fc  = nn.Linear(self.flat_feats, 100)
        self.bn  = nn.BatchNorm1d(100)	
	self.re  = nn.ReLU(inplace=True)
        self.fc_out  = nn.Linear(100, num_classes)
	
        ## Weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d or nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d or nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.expansion(x)
	## block 1
        x = self.cnn1(self.mp1(self.block1(x)))
	#print(x.size())
        ## block 2
        x = self.cnn2(self.mp2(self.block2(x)))
        #print(x.size())
        ## block 3
        x = self.cnn3(self.mp3(self.block3(x)))
        #print(x.size())
        ## block 4
        x = self.cnn4(self.mp4(self.block4(x)))
	#print(x.size())
        ## FC
	x = self.fc_out(self.re(self.bn(self.fc(x.view(-1, self.flat_feats)))))
        #print(x.size())
 
        if self.binary: return x 
        else: return F.log_softmax(x, dim=-1) # take log-softmax over C classes

class SpoofSmallResNet257_400(nn.Module):
    ''' small ResNet (less GPU memory) for 257 by 400 feature map '''
    def __init__(self, num_classes, resnet_blocks=1, focal_loss=False, input_size=(1,257,400)):

        super(SpoofSmallResNet257_400, self).__init__()
        
        self.focal_loss = focal_loss

        self.expansion = nn.Conv2d(1, 8, kernel_size=(3,3), padding=(1,1))
        ## block 1
        layers = []
        for i in range(resnet_blocks):
            layers.append(ResBasicBlock(8))
        self.block1 = nn.Sequential(*layers)
        self.mp1    = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn1   = nn.Conv2d(8, 16, kernel_size=(3,3), dilation=(2,2))
        ## block 2
	layers = []
        for i in range(resnet_blocks):
            layers.append(ResBasicBlock(16))
        self.block2 = nn.Sequential(*layers)
        self.mp2    = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn2   = nn.Conv2d(16, 32, kernel_size=(3,3), dilation=(4,4))
        ## block 3
	layers = []
        for i in range(resnet_blocks):
            layers.append(ResBasicBlock(32))
        self.block3 = nn.Sequential(*layers)
        self.mp3    = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn3   = nn.Conv2d(32, 64, kernel_size=(3,3), dilation=(4,4))
        ## block 4
	layers = []
        for i in range(resnet_blocks):
            layers.append(ResBasicBlock(64))
        self.block4 = nn.Sequential(*layers)
        self.mp4    = nn.MaxPool2d(kernel_size=(1,2))
        self.cnn4   = nn.Conv2d(64, 64, kernel_size=(3,3), dilation=(8,8)) 

        self.flat_feats = 64*3*2

        self.fc  = nn.Linear(self.flat_feats, 100)
        self.bn  = nn.BatchNorm1d(100)	
	self.re  = nn.ReLU(inplace=True)
        self.fc_out  = nn.Linear(100, num_classes)
	
        ## Weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d or nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d or nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.expansion(x)
	## block 1
        x = self.cnn1(self.mp1(self.block1(x)))
	#print(x.size())
        ## block 2
        x = self.cnn2(self.mp2(self.block2(x)))
        #print(x.size())
        ## block 3
        x = self.cnn3(self.mp3(self.block3(x)))
        #print(x.size())
        ## block 4
        x = self.cnn4(self.mp4(self.block4(x)))
	#print(x.size())
        ## FC
	x = self.fc_out(self.re(self.bn(self.fc(x.view(-1, self.flat_feats)))))
        #print(x.size())
 
        if self.focal_loss: return x
        else: return F.log_softmax(x, dim=-1) # take log-softmax over C classes

class SpoofResNet30_400(nn.Module):
    ''' primative ResNet for 30 by 400 feature map '''
    def __init__(self, num_classes, resnet_blocks=1, input_size=(1,30,400)):

        super(SpoofResNet30_400, self).__init__()
        
        self.expansion = nn.Conv2d(1, 16, kernel_size=(3,3), padding=(1,1))
        ## block 1
        layers = []
        for i in range(resnet_blocks):
            layers.append(ResBasicBlock(16))
        self.block1 = nn.Sequential(*layers)
        self.mp1    = nn.MaxPool2d(kernel_size=(1,1))
        self.cnn1   = nn.Conv2d(16, 32, kernel_size=(3,3), dilation=(1,2))
        ## block 2
	layers = []
        for i in range(resnet_blocks):
            layers.append(ResBasicBlock(32))
        self.block2 = nn.Sequential(*layers)
        self.mp2    = nn.MaxPool2d(kernel_size=(1,1))
        self.cnn2   = nn.Conv2d(32, 32, kernel_size=(3,3), dilation=(2,4))
        ## block 3
	layers = []
        for i in range(resnet_blocks):
            layers.append(ResBasicBlock(32))
        self.block3 = nn.Sequential(*layers)
        self.mp3    = nn.MaxPool2d(kernel_size=(1,2))
        self.cnn3   = nn.Conv2d(32, 32, kernel_size=(3,3), dilation=(2,4))
        ## block 4
	layers = []
        for i in range(resnet_blocks):
            layers.append(ResBasicBlock(32))
        self.block4 = nn.Sequential(*layers)
        self.mp4    = nn.MaxPool2d(kernel_size=(1,2))
        self.cnn4   = nn.Conv2d(32, 32, kernel_size=(3,3), dilation=(4,8)) 
        ## block 5
	layers = []
        for i in range(resnet_blocks):
            layers.append(ResBasicBlock(32))
        self.block5 = nn.Sequential(*layers)
        self.mp5    = nn.MaxPool2d(kernel_size=(1,4))
        self.cnn5   = nn.Conv2d(32, 32, kernel_size=(3,3), dilation=(4,8)) 

        self.flat_feats = 32*4*3        

        self.fc  = nn.Linear(self.flat_feats, 100)
        self.bn  = nn.BatchNorm1d(100)	
	self.re  = nn.ReLU(inplace=True)
        self.fc_out  = nn.Linear(100, num_classes)
	
        ## Weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d or nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d or nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.expansion(x)
	## block 1
        x = self.cnn1(self.mp1(self.block1(x)))
	#print(x.size())
        ## block 2
        x = self.cnn2(self.mp2(self.block2(x)))
        #print(x.size())
        ## block 3
        x = self.cnn3(self.mp3(self.block3(x)))
        #print(x.size())
        ## block 4
        x = self.cnn4(self.mp4(self.block4(x)))
	#print(x.size())
        ## block 5
        x = self.cnn5(self.mp5(self.block5(x)))
        #print(x.size())
        ## FC
	x = self.fc_out(self.re(self.bn(self.fc(x.view(-1, self.flat_feats)))))
        #print(x.size())
 
        return F.log_softmax(x, dim=-1) # take log-softmax over C classes

    def predict(self, x):
        raise NotImplementedError()

class SpoofResNet257_500(nn.Module):
    ''' primiative ResNet for feature map 257 by 500 '''
    def __init__(self, num_classes, resnet_blocks=1, input_size=(1,257,500)):

        super(SpoofResNet257_500, self).__init__()
        
        self.expansion = nn.Conv2d(1, 16, kernel_size=(3,3), padding=(1,1))
        ## block 1
        layers = []
        for i in range(resnet_blocks):
            layers.append(ResBasicBlock(16))
        self.block1 = nn.Sequential(*layers)
        self.mp1    = nn.MaxPool2d(kernel_size=(1,1))
        self.cnn1   = nn.Conv2d(16, 32, kernel_size=(3,3), dilation=(2,2))
        ## block 2
	layers = []
        for i in range(resnet_blocks):
            layers.append(ResBasicBlock(32))
        self.block2 = nn.Sequential(*layers)
        self.mp2    = nn.MaxPool2d(kernel_size=(1,1))
        self.cnn2   = nn.Conv2d(32, 32, kernel_size=(3,3), dilation=(4,4))
        ## block 3
	layers = []
        for i in range(resnet_blocks):
            layers.append(ResBasicBlock(32))
        self.block3 = nn.Sequential(*layers)
        self.mp3    = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn3   = nn.Conv2d(32, 32, kernel_size=(3,3), dilation=(4,4))
        ## block 4
	layers = []
        for i in range(resnet_blocks):
            layers.append(ResBasicBlock(32))
        self.block4 = nn.Sequential(*layers)
        self.mp4    = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn4   = nn.Conv2d(32, 32, kernel_size=(3,3), dilation=(8,8)) 
        ## block 5
	layers = []
        for i in range(resnet_blocks):
            layers.append(ResBasicBlock(32))
        self.block5 = nn.Sequential(*layers)
        self.mp5    = nn.MaxPool2d(kernel_size=(2,4))
        self.cnn5   = nn.Conv2d(32, 32, kernel_size=(3,3), dilation=(8,11)) # change dilation rate from (8,8) to (8,11) 

        self.flat_feats = 32*4*3        

        self.fc  = nn.Linear(self.flat_feats, 100)
        self.bn  = nn.BatchNorm1d(100)	
	self.re  = nn.ReLU(inplace=True)
        self.fc_out  = nn.Linear(100, num_classes)
	
        ## Weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d or nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d or nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        ##print(x.size())
        x = self.expansion(x)
	## block 1
        x = self.cnn1(self.mp1(self.block1(x)))
	##print(x.size())
        ## block 2
        x = self.cnn2(self.mp2(self.block2(x)))
        ##print(x.size())
        ## block 3
        x = self.cnn3(self.mp3(self.block3(x)))
        ##print(x.size())
        ## block 4
        x = self.cnn4(self.mp4(self.block4(x)))
	##print(x.size())
        ## block 5
        x = self.cnn5(self.mp5(self.block5(x)))
        ##print(x.size())
        ## FC
	x = self.fc_out(self.re(self.bn(self.fc(x.view(-1, self.flat_feats)))))
        ##print(x.size())
 
        return F.log_softmax(x, dim=-1) # take log-softmax over C classes

    def predict(self, x):
        raise NotImplementedError()

class SpoofResNet257_400(nn.Module):
    ''' primative ResNet for 257 by 400 feature map '''
    def __init__(self, num_classes, resnet_blocks=1, input_size=(1,257,400)):

        super(SpoofResNet257_400, self).__init__()
        
        self.expansion = nn.Conv2d(1, 16, kernel_size=(3,3), padding=(1,1))
        ## block 1
        layers = []
        for i in range(resnet_blocks):
            layers.append(ResBasicBlock(16))
        self.block1 = nn.Sequential(*layers)
        self.mp1    = nn.MaxPool2d(kernel_size=(1,1))
        self.cnn1   = nn.Conv2d(16, 32, kernel_size=(3,3), dilation=(2,2))
        ## block 2
	layers = []
        for i in range(resnet_blocks):
            layers.append(ResBasicBlock(32))
        self.block2 = nn.Sequential(*layers)
        self.mp2    = nn.MaxPool2d(kernel_size=(1,1))
        self.cnn2   = nn.Conv2d(32, 32, kernel_size=(3,3), dilation=(4,4))
        ## block 3
	layers = []
        for i in range(resnet_blocks):
            layers.append(ResBasicBlock(32))
        self.block3 = nn.Sequential(*layers)
        self.mp3    = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn3   = nn.Conv2d(32, 32, kernel_size=(3,3), dilation=(4,4))
        ## block 4
	layers = []
        for i in range(resnet_blocks):
            layers.append(ResBasicBlock(32))
        self.block4 = nn.Sequential(*layers)
        self.mp4    = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn4   = nn.Conv2d(32, 32, kernel_size=(3,3), dilation=(8,8)) 
        ## block 5
	layers = []
        for i in range(resnet_blocks):
            layers.append(ResBasicBlock(32))
        self.block5 = nn.Sequential(*layers)
        self.mp5    = nn.MaxPool2d(kernel_size=(2,4))
        self.cnn5   = nn.Conv2d(32, 32, kernel_size=(3,3), dilation=(8,8)) 

        self.flat_feats = 32*4*3        

        self.fc  = nn.Linear(self.flat_feats, 100)
        self.bn  = nn.BatchNorm1d(100)	
	self.re  = nn.ReLU(inplace=True)
        self.fc_out  = nn.Linear(100, num_classes)
	
        ## Weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d or nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d or nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.expansion(x)
	## block 1
        x = self.cnn1(self.mp1(self.block1(x)))
	##print(x.size())
        ## block 2
        x = self.cnn2(self.mp2(self.block2(x)))
        ##print(x.size())
        ## block 3
        x = self.cnn3(self.mp3(self.block3(x)))
        ##print(x.size())
        ## block 4
        x = self.cnn4(self.mp4(self.block4(x)))
	##print(x.size())
        ## block 5
        x = self.cnn5(self.mp5(self.block5(x)))
        ##print(x.size())
        ## FC
	x = self.fc_out(self.re(self.bn(self.fc(x.view(-1, self.flat_feats)))))
        ##print(x.size())
 
        return F.log_softmax(x, dim=-1) # take log-softmax over C classes

    def predict(self, x):
        raise NotImplementedError()

