import torch
import torch.nn as nn
from torch.autograd import Variable

from capsule_layer import CapsuleLayer
from convrelu import Conv2dRelu


class CapsNet(nn.Module):
    def __init__(self, n_conv_channel, n_primary_caps, primary_cap_size,
                 output_unit_size, n_routing_caps, use_gpu):

        super(CapsNet, self).__init__()

        self.use_gpu = use_gpu
        if use_gpu:
            self.cuda()

        self.conv1 = Conv2dRelu(1, n_conv_channel, 9)

        self.primary = CapsuleLayer(0,
                                    n_conv_channel,
                                    n_primary_caps,
                                    primary_cap_size,
                                    False,
                                    n_routing_caps,
                                    use_gpu)

        self.routing_caps = CapsuleLayer(n_primary_caps,
                                         primary_cap_size,
                                         10,  # 10 catagories in MNIST
                                         output_unit_size,
                                         True,
                                         n_routing_caps,
                                         use_gpu)

    def forward(self, X):
        X = self.conv1(X)
        X = self.primary(X)
        X = self.routing_caps(X)
        return X

    def loss(self, input, target, size_average=True):
        """Custom loss function"""
        m_loss = self.margin_loss(input, target, size_average)
        return m_loss

    def margin_loss(self, input, target, size_average=True):
        """Margin loss for digit existence
        """
        batch_size = input.size(0)

        # Implement equation 4 in the paper.

        # ||vc||
        v_c = torch.sqrt((input**2).sum(dim=2, keepdim=True))

        # Calculate left and right max() terms.
        zero = Variable(torch.zeros(1))
        if self.use_gpu:
            zero = zero.cuda()
        m_plus = 0.9
        m_minus = 0.1
        loss_lambda = 0.5
        max_left = torch.max(m_plus - v_c, zero).view(batch_size, -1)
        max_right = torch.max(v_c - m_minus, zero).view(batch_size, -1)
        t_c = target
        # Lc is margin loss for each digit of class c
        l_c = t_c * max_left + loss_lambda * (1.0 - t_c) * max_right
        l_c = l_c.sum(dim=1)

        if size_average:
            l_c = l_c.mean()

        return l_c
