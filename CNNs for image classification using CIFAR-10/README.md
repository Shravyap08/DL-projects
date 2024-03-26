# CNNs for image classification using CIFAR-10


## Project Overview
This project involves the implementation of a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset. The model consists of four convolutional blocks with batch normalization, max-pooling, and ReLU activation functions, followed by two fully connected layers for classification.

## Model Design

### Convolutional Blocks
- **First Block**: 3 input channels with 16 output channels, batch normalization, 2x2 max pooling, ReLU activation.
- **Second Block**: 16 input channels with 32 output channels, batch normalization, 2x2 max pooling (stride=1), ReLU activation.
- **Third Block**: 32 input channels with 64 output channels (3x3 kernel), batch normalization, ReLU activation, 2x2 max pooling (stride=1).
- **Fourth Block**: 64 input channels with 128 output channels (3x3 kernel), batch normalization, ReLU activation, 2x2 max pooling (stride=1).

### Fully Connected Layers
- **First Layer**: Linear layer with 15488 input features and 128 output features, batch normalization, ReLU activation.
- **Second Layer**: Linear layer with 128 inputs and 10 outputs (number of classes).

### Justification
Batch normalization is employed to enhance model stability and speed up training. ReLU activation functions are chosen for computational efficiency, while max pooling reduces dimensionality and computational complexity. Multiple blocks allow the model to learn a variety of robust features.

## Hyperparameters

- **Batch Size**: 32
- **Learning Rate**: 0.01
- **Regularization**: 0.0005
- **Epochs**: 10
- **Momentum**: 0.9

These hyperparameters strike a balance between optimization, gradient estimation accuracy, model learning speed, regularization strength, and momentum typicality for many applications.

## Results

Final accuracy on the validation set: **82.01%**

### Training with Regular CE Loss on Imbalanced CIFAR-10
Per-class accuracy results indicate strong performance on majority classes with CE Loss, while minority classes suffered due to data imbalance.

### Training with CB-Focal Loss on Imbalanced CIFAR-10
The CB-Focal Loss significantly improved the performance on minority classes without compromising majority class accuracy, demonstrating the effectiveness of the re-weighting strategy in focal loss for dealing with imbalanced datasets.

Best PREC @1 Accuracy:
- **CB Focal Loss**: 0.3648
- **CE Loss**: 0.3480

## Testing and Verification
Unit tests were conducted to verify the functionality of focal loss compared to standard cross-entropy by setting gamma=0, ensuring values matched as expected. As gamma values increased, the tests confirmed the focal loss's sensitivity, proving its correctness.

## Conclusion
The model's robust architecture, along with the careful selection of hyperparameters, achieves a balance between learning capacity and computational efficiency. The results demonstrate the superiority of CB-Focal Loss in addressing class imbalance issues within the CIFAR-10 dataset.

## Acknowledgments
This project was conducted as part of an advanced machine learning course at [University Name/Institution]. Special thanks to the instructors and colleagues for their valuable feedback and insights.

