import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class CapsuleLayer(nn.Module):

    def __init__(self, n_primary_caps, in_channels, n_unit, unit_size,
                 routing_cap, n_routing, use_gpu=False):
        '''
        params:
            n_primary_caps (int): only needed if this is are routing cap. number of primary capsules to expect
            in_channels (int): number of channels on the tensor fed into capsule
            n_units (int): number of capsule units
            unit_size (int): size of each capsule
            routing_cap (bool): if this layer of caps is primary or routing (final)
            n_routing (int): if a routing layer, how many iters to run the routing alg
            use_gpu (bool): whether or not to move model to gpu
        '''
        super(CapsuleLayer, self).__init__()

        self.n_primary_caps = n_primary_caps
        self.in_channels = in_channels
        self.n_unit = n_unit
        self.unit_size = unit_size
        self.routing_cap = routing_cap
        self.n_routing = n_routing
        self.use_gpu = use_gpu

        if self.routing_cap:
            self.W = nn.Parameter(torch.randn(1, in_channels, n_unit,
                                              unit_size, n_primary_caps))

        else:
            self.unit_convs = []
            for i in range(self.n_unit):
                # >in_channels, 32 out, kernel of 9, stride of 2
                c = nn.Conv2d(in_channels, 32, 9, 2)

                self.add_module('conv_unit_{}'.format(i), c)
                self.unit_convs.append(c)

    def squash(self, sj):
        """
        Non-linear 'squashing' function.
        This implement equation 1 from the paper.
        """
        sj_mag_sq = torch.sum(sj**2, dim=2, keepdim=True)
        # ||sj ||
        sj_mag = torch.sqrt(sj_mag_sq)
        v_j = (sj_mag_sq / (1.0 + sj_mag_sq)) * (sj / sj_mag)
        return v_j

    def forward(self, X):
        if self.routing_cap:
            return self.routing(X)

        else:
            unit = [self.unit_convs[i](X) for i in range(self.n_unit)]
            unit = torch.stack(unit, dim=1)
            unit = unit.view(X.size(0), self.n_unit, -1)
            return self.squash(unit)

    def routing(self, X):
        batch_size = X.size(0)

        X = X.transpose(1, 2)
        X = torch.stack([X] * self.n_unit, dim=2).unsqueeze(4)

        W = torch.cat([self.W] * batch_size)

        # apply weights
        u_hat = torch.matmul(W, X)

        b_ij = Variable(torch.zeros(1, self.in_channels, self.n_unit, 1))
        b_ij = b_ij.cuda() if self.use_gpu else b_ij

        for i in range(self.n_routing):
            # Routing described in Precedure 1
            c_ij = F.softmax(b_ij)
            c_ij = torch.cat([c_ij] * batch_size).unsqueeze(4)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)

            v_j = self.squash(s_j)

            v_j1 = torch.cat([v_j] * self.in_channels, dim=1)

            u_vj1 = torch.matmul(u_hat.transpose(3, 4), v_j1)
            u_vj1 = u_vj1.squeeze(4).mean(dim=0, keepdim=True)

            b_ij = u_vj1 + b_ij

        return v_j.squeeze(1)


