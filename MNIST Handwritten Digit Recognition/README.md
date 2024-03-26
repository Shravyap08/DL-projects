# MNIST Handwritten Digit Recognition

## Overview
This project focuses on building and training deep neural networks to recognize MNIST Handwritten Digits. Implementation of two neural network architectures: 
Softmax Regression and a 2-layer Multi-Layer Perceptron (MLP). The project covers the entire pipeline, from data loading and preprocessing to model training, optimization, and evaluation.

## Libraries and Frameworks

This project leverages several Python libraries and frameworks to implement and train the neural network models for digit recognition on the MNIST dataset. Below is a list of the main libraries used:

- **PyTorch**
- **NumPy**
- **Matplotlib**
- **Pandas**
- **Scikit-learn**

## Key Findings

### Learning Rates

Conducted experiments with different learning rates while keeping other hyper-parameters fixed. Our findings indicate that a higher learning rate tends to achieve better accuracy, but too high of a learning rate might lead to instability in training.

#### Results:

| Learning Rate | Training Accuracy | Test Accuracy |
|---------------|-------------------|---------------|
| 1             | 0.9524            | 0.9519        |
| 1e-1          | 0.9270            | 0.9265        |
| 5e-2          | 0.9158            | 0.9135        |
| 1e-2          | 0.7728            | 0.7615        |

### Regularization

Adjusting the regularization coefficient showed that lower values lead to better model performance, suggesting that too much regularization may hinder the model's ability to learn complex patterns.

#### Results:

| Regularization Coefficient | Training Accuracy | Validation Accuracy | Test Accuracy |
|----------------------------|-------------------|---------------------|---------------|
| 1                          | 0.1047            | 0.1035              | 0.1028        |
| 1e-1                       | 0.3906            | 0.3750              | 0.3603        |
| 1e-2                       | 0.9219            | 0.8949              | 0.8915        |
| 1e-3                       | 0.9372            | 0.9265              | 0.9260        |
| 1e-4                       | 0.9374            | 0.9352              | 0.9340        |

### Hyper-parameter Tuning

Through comprehensive tuning, we identified an optimal model configuration that balances the training and test accuracy, preventing overfitting while achieving high performance.

#### Best Configuration:

- **Regulation:** 1e-4
- **Learning Rate:** 1
- **Batch Size:** 64
- **Training Accuracy:** 0.9844
- **Validation Accuracy:** 0.9697
- **Test Accuracy:** 0.9711

## Conclusion

All experiments underscore the importance of carefully choosing hyper-parameters for training neural networks. The optimal configuration achieved high accuracy without overfitting, highlighting the effectiveness of our model on the MNIST dataset.

## Dataset Source

The MNIST Handwritten Digit Dataset used in this project can be found [here](http://yann.lecun.com/exdb/mnist/).
