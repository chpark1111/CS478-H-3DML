import torch
import torch.nn  as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, ):
        super(Encoder, self).__init__()
        

    def forward(self, x):
        """
        Input: (batch_size, n_points, in_dim).
        Outputs: (batch_size, n_classes).
        """
        
        return x


class Decoder(nn.Module):
    def __init__(self, ):
        super(Decoder, self).__init__()
        

    def forward(self, x):
        """
        Input: (batch_size, n_points, in_dim).
        Outputs: (batch_size, n_classes).
        """

        return x


class CSGmodel(nn.Module):
    def __init__(self,):
        super(CSGmodel, self).__init__()
        self.encoder = Encoder()

    def forward(self, x):
        """
        Input: (batch_size, n_points, in_dim).
        Outputs: (batch_size, n_classes).
        """

        return x