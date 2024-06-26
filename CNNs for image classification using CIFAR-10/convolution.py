"""
2d Convolution Module.  (c) 2021 Georgia Tech

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

import numpy as np


class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        """
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        """
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        """
        out = None
        
       
        S=self.stride
        P=self.padding
        KH= self.kernel_size
        KW= self.kernel_size
        N,C,H,W= x.shape
       
        H_O= int((H-KH+(2*P))/S)+1
        W_O= int((W-KW+(2*P))/S)+1
        C_O = self.out_channels
        out = np.zeros((N, C_O, H_O, W_O))
        
        xpad=np.pad(x, ((0,0),(0,0),(P,P),(P,P)),'constant')
        
       
        for n in range(N):
            for c_o in range(C_O):
                for h in range(H_O):
                    for w in range(W_O):
                        h1 = h * S
                        h2 = h1 + KH
                        w1 = w * S
                        w2 = w1 + KW
                    
                   
                        x_piece=xpad[n,:,h1:h2,w1:w2]
                        out[n, c_o, h, w]=np.sum(x_piece*self.weight[c_o])+self.bias[c_o]
       
                                         
        
       #############################################################################
        # TODO: Implement the convolution forward pass.                             #
        # Hint: 1) You may use np.pad for padding.                                  #
        #       2) You may implement the convolution with loops                     #
        #############################################################################

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = x,xpad,out
        return out

    def backward(self, dout):
        """
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        """
        
        
        x,xpad,out=self.cache
        N,C,H,W= x.shape
       
        S=self.stride
        P=self.padding
        KH= self.kernel_size
        KW= self.kernel_size
        
        C = self.weight.shape[0]
        H_out, W_out = out.shape[2], out.shape[3]
        
        
        dx=np.zeros_like(x)
        dx_pad = np.zeros_like(xpad)
        dw=np.zeros_like(self.weight)
        db=np.zeros_like(self.bias)
        
        
        
        for n in range(N):
            for c in range(C):
                for h in range(0, H_out * S, S):
                    for w in range(0, W_out * S, S):
                        dw[c] += xpad[n, :, h:h+KH, w:w+KW] * dout[n, c, h//S, w//S]
                        db[c] += dout[n, c, h//S, w//S]
                        dx_pad[n, :, h:h+KH, w:w+KW] += self.weight[c] * dout[n, c, h//S, w//S]
                        
        if P != 0:
            dx = dx_pad[:, :, P:-P, P:-P]
        else:
            dx = dx_pad
            
        self.dx = dx
        self.dw = dw
        self.db = db
        
       
        

        
        
        
        
      
                            
                        
        
        
        
        
        
        
        
        #############################################################################
        # TODO: Implement the convolution backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the convolution with loops                     #
        #       2) don't forget padding when computing dx                           #
        #############################################################################

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
