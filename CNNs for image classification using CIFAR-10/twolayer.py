"""
Two Layer Network Model.  (c) 2021 Georgia Tech

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


class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes):
        """
        :param input_dim: input feature dimension
        :param hidden_size: hidden dimension
        :param num_classes: total number of classes
        """
        super(TwoLayerNet, self).__init__()
        self.layer1= nn.Linear(input_dim,hidden_size)
        #act_func=torch.sigmoid(self.layer1)
        self.layer2= nn.Linear(hidden_size,num_classes)
        
        #############################################################################
        # TODO: Initialize the TwoLayerNet, use sigmoid activation between layers   #
        #############################################################################

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        out = None
        
        x=x.view(x.size(0), -1)
        
        y=self.layer1(x)
        sig_out=torch.sigmoid(y)
        out= self.layer2(sig_out)
        
        
        
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out
