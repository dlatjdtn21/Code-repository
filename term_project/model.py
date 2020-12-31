import torch.nn as nn
import torch.nn.functional as Func


""" Optional conv block """
def conv_block(in_channels, out_channels):

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout(p=0.5)
    )


""" Define your own model """
class FewShotModel(nn.Module):
    def __init__(self, x_dim=3, hid_dim1=16, hid_dim2=32 ,hid_dim3=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim1),
            conv_block(hid_dim1, hid_dim1),
            conv_block(hid_dim1, hid_dim2),
            conv_block(hid_dim2, hid_dim2),
            conv_block(hid_dim2, hid_dim3),
            conv_block(hid_dim3, hid_dim3),
            conv_block(hid_dim3, z_dim),
        )

    def forward(self, z):
        z = self.encoder(z)
        z = nn.MaxPool2d(2)(z)
        z = nn.Dropout(0.5)(z)
        embedding_vector = z.view(z.size(0), -1)
        return embedding_vector
