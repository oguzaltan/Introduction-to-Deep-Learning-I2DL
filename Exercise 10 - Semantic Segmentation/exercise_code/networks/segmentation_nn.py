"""SegmentationNN"""
import torch
import torch.nn as nn
from torchvision import models

class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23, hp=None):
        super().__init__()
        self.hp = hp

        self.img_width, self.img_height = self.hp['img_width'], self.hp['img_height']
        
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        
        # Load pre-trained AlexNet model
        # alexnet = models.alexnet(pretrained=True)
        resnet = models.resnet18(pretrained=True)

        # Extract the features from the pre-trained ResNet18
        # Remove the fully connected layers (classification head)
        self.features = nn.Sequential(*list(resnet.children())[:-2])

        # Adjust the classifier for your segmentation task
        self.classifier = nn.Sequential(
            
            # ResNet18's feature extractor output channel size is 512
            # Apply 1x1 convolution to decrease this output channel size to number of classes
            nn.Conv2d(512, num_classes, kernel_size=1), 
            
            # Upsample to the desired size
            # TODO: How to upconvolve using transposal convolution layers instead of upsampling?
            nn.Upsample(size=(self.img_width, self.img_height), mode='bilinear', align_corners=False),  
            )

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        
        # Forward pass through the feature extractor
        x = self.features(x)
        
        # Forward pass through the adjusted classifier
        x = self.classifier(x)
        
        return x
    
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    # @property
    # def is_cuda(self):
    #     """
    #     Check if model parameters are allocated on the GPU.
    #     """
    #     return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
class DummySegmentationModel(nn.Module):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()

if __name__ == "__main__":
    from torchinfo import summary
    summary(SegmentationNN(), (1, 3, 240, 240), device="cpu")