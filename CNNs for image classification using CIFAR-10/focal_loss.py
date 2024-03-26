"""
Focal Loss Wrapper.  (c) 2021 Georgia Tech

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
import torch.nn.functional as F
import numpy as np


def reweight(cls_num_list, beta=0.9999):
    """
    Implement reweighting by effective numbers
    :param cls_num_list: a list containing # of samples of each class
    :param beta: hyper-parameter for reweighting, see paper for more details
    :return:
    """
   
    per_cls_weights = None
    cls_num_list=np.array(cls_num_list)
    Effective_numbers= (1-np.power(beta,cls_num_list))
    per_cls_weights=(1-beta)/np.array(Effective_numbers)
    
    per_cls_weights= per_cls_weights/np.sum(per_cls_weights)*len(per_cls_weights)
    
    per_cls_weights = torch.tensor(per_cls_weights, dtype=torch.float32)
    
    #############################################################################
    # TODO: reweight each class by effective numbers                            #
    #############################################################################

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return per_cls_weights


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight
        
        

    def forward(self, input, target):
        """
        Implement forward of focal loss
        :param input: input predictions
        :param target: labels
        :return: tensor of focal loss in scalar
        """
        loss = None
        p = F.softmax(input, dim=1)
        pt = p[torch.arange(input.size(0)), target]
        loss = - (1 - pt).pow(self.gamma) * torch.log(pt)
        if self.weight is not None:
            loss = loss * self.weight[target]
            
        loss = loss.mean()

        return loss

        #loss = F.cross_entropy(input, target, self.weight)
        #loss=torch.mean(loss)
        
        #############################################################################
        # TODO: Implement forward pass of the focal loss                            #
        #############################################################################

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        
