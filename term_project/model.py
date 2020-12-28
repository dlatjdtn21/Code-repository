import torch.nn as nn
import torch.nn.functional as Func


""" Optional conv block """
def conv_block(in_channels, out_channels):

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout(p=0.4)
    )


""" Define your own model """
class FewShotModel(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super(FewShotModel, self).__init__()
        self.conv1 = conv_block(28, 64)
        self.conv2 = conv_block(64, 64)

    def forward(self, z):
        z = self.conv1(z)
        z = self.conv2(z)

        embedding_vector = z.view(z.size(0), -1)
        return embedding_vector
