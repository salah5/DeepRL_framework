# -*- coding: utf-8 -*-
import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F

class Attention(nn.Module):
    r"""
    Applies an attention mechanism on the output features from the decoder.

    .. math::
            \begin{array}{ll}
            x = context*output \\
            attn = exp(x_i - max_i x_i) / sum_j exp(x_j - max_i x_i) \\
            output = \tanh(w * (attn * context) + b * output)
            \end{array}

    Args:
        dim(int): The number of expected features in the output

    Inputs: output, context
        - **output** (batch, output_len, dimensions): tensor containing the output features from the decoder.
        - **context** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.

    Outputs: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn** (batch, output_len, input_len): tensor containing attention weights.

    Attributes:
        linear_out (torch.nn.Linear): applies a linear transformation to the incoming data: :math:`y = Ax + b`.
        mask (torch.Tensor, optional): applies a :math:`-inf` to the indices specified in the `Tensor`.

    Examples::

         >>> attention = Attention(256)
         >>> context = Variable(torch.randn(5, 3, 256))
         >>> output = Variable(torch.randn(5, 5, 256))
         >>> output, attn = attention(output, context)

    """

    def __init__(self, dec_size, attn_size, mode, project=False):
        super(Attention, self).__init__()
        self.mask = None
        self.mode = mode
        self.attn_size = attn_size
        self.dec_size = dec_size

        if project:
            self.linear_out = nn.Linear(dec_size+attn_size, dec_size)
        else:
            self.linear_out = None

        if mode == 'general':
            self.attn_w = nn.Linear(dec_size, attn_size)
        elif mode == 'cat':
            self.dec_w = nn.Linear(dec_size, dec_size)
            self.attn_w = nn.Linear(attn_size, dec_size)
            self.query_w = nn.Linear(dec_size, 1)

    def forward(self, output, context):
        """
        :param output: [batch, out_len, dec_size]
        :param context: [batch, in_len, attn_size]
        :return: output, attn
        """
        batch_size = output.size(0)
        input_size = context.size(1)

        # batch, out_len, in_len
        if self.mode == 'dot':
            attn = torch.bmm(output, context.transpose(1, 2))
        elif self.mode == 'general':
            mapped_output = self.attn_w(output)
            attn = torch.bmm(mapped_output, context.transpose(1, 2))
        elif self.mode == 'cat':
            mapped_attn = self.attn_w(context)
            mapped_out = self.dec_w(output)
            tiled_out = mapped_out.unsqueeze(2).repeat(1, 1, input_size, 1)
            tiled_attn = mapped_attn.unsqueeze(1)
            fc1 = F.tanh(tiled_attn+tiled_out)
            attn = self.query_w(fc1).squeeze(-1)

        else:
            raise ValueError("Unknown attention")

        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))

        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        # (batch, out_len, in_len) * (batch, in_len, dim)
        #  -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)
        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, output), dim=2)

        if self.linear_out is None:
            return combined, attn
        else:
            # output -> (batch, out_len, dim)
            output = F.tanh(
                self.linear_out(combined.view(-1, self.dec_size+self.attn_size))).view(
                batch_size, -1, self.dec_size)
            return output, attn

