""" 			  		 			     			  	   		   	  			  	
Softmax Regression Model.  (c) 2021 Georgia Tech

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

# Do not use packages that are not in standard distribution of python
import numpy as np

from ._base_network import _baseNetwork


class SoftmaxRegression(_baseNetwork):
    def __init__(self, input_size=28 * 28, num_classes=10):
        super().__init__(input_size, num_classes)
        self._weight_init()

    def _weight_init(self):
        np.random.seed(1024)
        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.num_classes)
        self.gradients['W1'] = np.zeros((self.input_size, self.num_classes))

    def forward(self, X, y, mode='train'):
        
        self.cache = {}
        
        # FORWARD PROCESS
        
        Z = np.dot(X, self.weights['W1'])
        self.cache['Z'] = Z
        
        # RELU 
        Z_relu = np.maximum(0, Z)
        
        
        # Softmax
        exp_Z = np.exp(Z_relu - np.max(Z_relu, axis=1, keepdims=True))
        p = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        loss = self.cross_entropy_loss(p, y)
        accuracy = self.compute_accuracy(p, y)
        

        if mode != 'train':
            return loss, accuracy

        # BACKWARD PASS
        N = X.shape[0]
        y_one_hot = np.zeros_like(p)
        y_one_hot[np.arange(N), y] = 1
        dZ = (p - y_one_hot) / N
        Z = self.cache['Z']
        dZ[Z <= 0] = 0  
        self.gradients['W1'] = np.dot(X.T, dZ)
        return loss, accuracy
