"""Models for facial keypoint detection"""

import torch
import torch.nn as nn

class KeypointModel(nn.Module):
    """Facial keypoint detection model"""
    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
            
        """
        super().__init__()
        self.hparams = hparams
        
        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        #                                                                      #
        # We would truly recommend to make your code generic, such as you      #
        # automate the calculation of the number of parameters at each layer.  #
        # You're going to probably try different architectures, and that will  #
        # allow you to be quick and flexible.                                  #
        ########################################################################
        
        # Define the convolutional layers 
        self.conv1 = nn.Conv2d(self.hparams['input_channel'], self.hparams['conv1_out_channel'], kernel_size=self.hparams['conv1_kernel_size'], padding=1)
        self.bn1 = nn.BatchNorm2d(self.hparams['conv1_out_channel'])
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=self.hparams['pooling_size'], stride=self.hparams['pooling_stride'])
        self.dropout1 = nn.Dropout(self.hparams['dropout_p'])

        self.conv2 = nn.Conv2d(self.hparams['conv1_out_channel'], self.hparams['conv2_out_channel'], kernel_size=self.hparams['conv2_kernel_size'], padding=1)
        self.bn2 = nn.BatchNorm2d(self.hparams['conv2_out_channel'])
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=self.hparams['pooling_size'], stride=self.hparams['pooling_stride'])
        self.dropout2 = nn.Dropout(self.hparams['dropout_p'])

        self.conv3 = nn.Conv2d(self.hparams['conv2_out_channel'], self.hparams['conv3_out_channel'], kernel_size=self.hparams['conv3_kernel_size'], padding=1)
        self.bn3 = nn.BatchNorm2d(self.hparams['conv3_out_channel'])
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=self.hparams['pooling_size'], stride=self.hparams['pooling_stride'])
        self.dropout3 = nn.Dropout(self.hparams['dropout_p'])
        
        # Define the fully connected layers
        self.fc1 = nn.Linear(int(self.hparams['conv3_out_channel']
                                * self.hparams['input_width'] / (self.hparams['pooling_stride'] ** self.hparams['number_of_conv_layers'])
                                * self.hparams['input_height'] / (self.hparams['pooling_stride'] ** self.hparams['number_of_conv_layers'])),
                                self.hparams['fc1_size'])
        # self.bn4 = nn.BatchNorm1d(self.hparams['fc1_size'])
        self.relu4 = nn.ReLU()
        # self.dropout4 = nn.Dropout(self.hparams['dropout_p'])
        self.fc2 = nn.Linear(self.hparams['fc1_size'], self.hparams['model_output_size'])  # Output layer with 30 keypoints
        
        
        ## Explicitly used parameter values
        
        # # Define the convolutional layers
        # self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm2d(32)
        # self.relu1 = nn.ReLU()
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.dropout1 = nn.Dropout(0.25)

        # self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(64)
        # self.relu2 = nn.ReLU()
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.dropout2 = nn.Dropout(0.25)

        # self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # self.bn3 = nn.BatchNorm2d(128)
        # self.relu3 = nn.ReLU()
        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.dropout3 = nn.Dropout(0.25)
        
        # # Define the fully connected layers
        # self.fc1 = nn.Linear(128 * 12 * 12, 200)
        # self.bn4 = nn.BatchNorm1d(200)
        # self.relu4 = nn.ReLU()
        # self.dropout4 = nn.Dropout(0.5)
        # self.fc2 = nn.Linear(200, 30)  # Output layer with 30 keypoints
        
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        
        # check dimensions to use show_keypoint_predictions later
        if x.dim() == 3:
            x = torch.unsqueeze(x, 0)
            
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints.                                   #
        # NOTE: what is the required output size?                              #
        ########################################################################

        # x = self.encoder(x)
                
        # Forward pass through convolutional layers
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool1(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool1(self.relu3(self.bn3(self.conv3(x))))

        # Reshape for fully connected layers
        x = x.view(x.size(0), -1)

        # Forward pass through fully connected layers
        # x = self.relu4(self.bn4(self.fc1(x)))
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return x


class DummyKeypointModel(nn.Module):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
