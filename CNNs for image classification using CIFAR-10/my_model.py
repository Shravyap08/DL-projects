"""
MyModel model.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1=nn.Conv2d(3, out_channels=16, kernel_size=3,padding=0)
        self.bn1 = nn.BatchNorm2d(16)  
        self.maxpool1=nn.MaxPool2d(2)
        self.relu=nn.ReLU()
        
        
        self.conv2=nn.Conv2d(16,out_channels=32, kernel_size=2, padding=0)
        self.bn2 = nn.BatchNorm2d(32)
        self.maxpool2=nn.MaxPool2d(kernel_size=2, stride=1)
        self.relu=nn.ReLU()
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.relu=nn.ReLU()
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.relu=nn.ReLU()
        
        
        
        self.fc1=nn.Linear(15488,128)
        self.bn5 =nn.BatchNorm1d(128)
        self.relu=nn.ReLU()

        self.fc2 = nn.Linear(128, 10)

        #############################################################################
        # TODO: Initialize the network weights                                      #
        #############################################################################

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        outs = None
        
        
        x = x.view(-1, 3, 32, 32)

        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool1(x)
        
        x=self.conv2(x)
        x = self.bn2(x)
        x=self.relu(x)
        x=self.maxpool2(x)
        
        x=self.conv3(x)
        x = self.bn3(x)
        x=self.relu(x)
        x=self.maxpool3(x)
        
        x=self.conv4(x)
        x = self.bn4(x)
        x=self.relu(x)
        x=self.maxpool4(x)
        
        x=x.view(x.size(0), -1)
        x=self.fc1(x)
        x=self.bn5(x)
        x=self.relu(x)
        
        x=self.fc2(x)
       
        outs=x
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outs
