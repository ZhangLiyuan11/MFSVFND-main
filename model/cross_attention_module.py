import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNormalization(nn.Module):
    def __init__(self, d_hid, eps=1e-6):
        super(LayerNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_hid))
        self.beta = nn.Parameter(torch.zeros(d_hid))
        self.eps = eps

    def forward(self, z):
        mean = z.mean(dim=-1, keepdim=True, )
        std = z.std(dim=-1, keepdim=True, )
        ln_out = (z - mean) / (std + self.eps)
        ln_out = self.gamma * ln_out + self.beta

        return ln_out

# from tools import *
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PoswiseFeedForwardNet, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, inputs):
        # inputs: [b_size x len_q x d_model]
        residual = inputs
        output = self.relu(self.conv1(inputs.transpose(1, 2)))

        # outputs: [b_size x len_q x d_model]
        output = self.conv2(output).transpose(1, 2)
        output = self.dropout(output)

        return self.layer_norm(residual + output)

class TransformerMLP(nn.Module):
    """transformer MLP in pytorch."""
    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            dropout=0.1
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(hidden_size)

    def forward(self, x):
        gate_inp = self.layer_norm(x)
        gate = self.gate_proj(gate_inp)
        gate = F.relu(gate)
        outputs = self.dropout(self.down_proj(gate))
        return self.layer_norm(outputs + x)

class CrossTransformer(nn.Module):
    def __init__(self, model_dimension, number_of_heads, dropout_probability):
        super().__init__()
        encoder_layer = EncoderLayer(model_dimension, dropout_probability, number_of_heads)
        self.encoder = Encoder(encoder_layer)
        self.init_params()

    def init_params(self, default_initialization=False):
        if not default_initialization:
            # model.named_parameters
            for name, p in self.named_parameters():
                if p.dim() > 1:
                    # 初始化均匀分布的网络参数
                    nn.init.xavier_uniform_(p)

    def forward(self, input1, input2):
        src_representations_batch = self.encoder(input1, input2)
        return src_representations_batch


class Encoder(nn.Module):
    def __init__(self, encoder_layer):
        super().__init__()
        assert isinstance(encoder_layer, EncoderLayer), f'Expected EncoderLayer got {type(encoder_layer)}.'
        self.encoder_layer = encoder_layer
        # self.ffn_layer = PoswiseFeedForwardNet(encoder_layer.model_dimension, encoder_layer.model_dimension * 2)
        self.ffn_layer = TransformerMLP(encoder_layer.model_dimension, encoder_layer.model_dimension * 2)
    def forward(self, src1, src2):
        # Forward pass through the encoder stack
        representations_outp = self.encoder_layer(src1, src2)
        representations = self.ffn_layer(representations_outp)
        return representations


class EncoderLayer(nn.Module):
    def __init__(self, model_dimension, dropout_probability, number_of_heads):
        super().__init__()
        self.sublayer1 = SublayerLogic(model_dimension, dropout_probability)
        self.sublayer2 = SublayerLogic(model_dimension, dropout_probability)
        self.mha1 = MultiHeadedAttention(model_dimension=model_dimension, number_of_heads=number_of_heads,
                                         dropout_probability=dropout_probability)
        self.mha2 = MultiHeadedAttention(model_dimension=model_dimension, number_of_heads=number_of_heads,
                                         dropout_probability=dropout_probability)
        self.model_dimension = model_dimension
        self.norm = nn.LayerNorm(model_dimension)

        self.fusion_mha = MultiHeadedAttention(model_dimension=model_dimension, number_of_heads=number_of_heads,
                                               dropout_probability=dropout_probability)
        self.fusion_sublayer = SublayerLogic(model_dimension, dropout_probability)

    def forward(self, srb1, srb2):
        attention1 = lambda srb1, srb2: self.mha1(query=srb1, key=srb2, value=srb2)
        attention2 = lambda srb1, srb2: self.mha2(query=srb1, key=srb2, value=srb2)
        representations_outp1 = self.norm(self.sublayer1(srb1, srb2, attention1))
        representations_outp2 = self.norm(self.sublayer2(srb2, srb1, attention2))
        fusion_rep = torch.cat((representations_outp1,representations_outp2),dim=1)
        fusion_attention = lambda srb1, srb2: self.fusion_mha(query=srb1, key=srb2, value=srb2)
        representations_outp = self.norm(self.fusion_sublayer(fusion_rep, fusion_rep, fusion_attention))

        return representations_outp


class SublayerLogic(nn.Module):
    def __init__(self, model_dimension, dropout_probability):
        super().__init__()
        self.norm = nn.LayerNorm(model_dimension)
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, srb1, srb2, mha):
        return srb1 + self.dropout(mha(self.norm(srb1), self.norm(srb2)))


class MultiHeadedAttention(nn.Module):
    def __init__(self, model_dimension, number_of_heads, dropout_probability):
        super().__init__()
        assert model_dimension % number_of_heads == 0, f'Model dimension must be divisible by the number of heads.'

        self.head_dimension = int(model_dimension / number_of_heads)
        self.number_of_heads = number_of_heads

        self.qkv_nets = get_clones(nn.Linear(model_dimension, model_dimension), 3)
        self.out_projection_net = nn.Linear(model_dimension, model_dimension)

        self.attention_dropout = nn.Dropout(p=dropout_probability)  # no pun intended, not explicitly mentioned in paper
        self.softmax = nn.Softmax(dim=-1)  # -1 stands for apply the softmax along the last dimension

        # self.log_attention_weights = log_attention_weights  # should we log attention weights
        self.attention_weights = None  # for visualization purposes, I cache the weights here (translation_script.py)

    def attention(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dimension)
        attention_weights = self.softmax(scores)
        attention_weights = self.attention_dropout(attention_weights)
        intermediate_token_representations = torch.matmul(attention_weights, value)

        return intermediate_token_representations, attention_weights  # attention weights for visualization purposes

    def forward(self, query, key, value):
        batch_size = query.shape[0]
        query, key, value = [net(x).view(batch_size, -1, self.number_of_heads, self.head_dimension).transpose(1, 2)
                             for net, x in zip(self.qkv_nets, (query, key, value))]
        intermediate_token_representations, attention_weights = self.attention(query, key, value)
        reshaped = intermediate_token_representations.transpose(1, 2).reshape(batch_size, -1,
                                                                              self.number_of_heads * self.head_dimension)
        # forward
        token_representations = self.out_projection_net(reshaped)
        return token_representations


def get_clones(module, num_of_deep_copies):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_of_deep_copies)])


# cross_trans = CrossTransformer(model_dimension=512,number_of_heads=8,dropout_probability=0.1)
# input1 = torch.randn(16,64,512)
# input2 = torch.randn(16,80,512)
#
# out = cross_trans(input1,input2)
# print('-----------')