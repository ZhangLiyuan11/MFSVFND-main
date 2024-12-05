import copy

import torch
from torch import nn
import torch.nn.functional as F
import math

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

class CausalAttention(nn.Module):
    def __init__(self, model_dimension, n_heads, dropout_probability=0.1):
        super(CausalAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = model_dimension

        self.query = nn.Linear(model_dimension, model_dimension)
        self.key = nn.Linear(model_dimension, model_dimension)
        self.value = nn.Linear(model_dimension, model_dimension)

        self.dropout = nn.Dropout(dropout_probability)
        self.scale = n_heads ** 0.5

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Linear projections
        Q = self.query(x).view(batch_size, seq_len, self.n_heads, self.d_model // self.n_heads)
        K = self.key(x).view(batch_size, seq_len, self.n_heads, self.d_model // self.n_heads)
        V = self.value(x).view(batch_size, seq_len, self.n_heads, self.d_model // self.n_heads)

        # Generate mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().unsqueeze(0).unsqueeze(1)
        mask = mask.expand(batch_size, self.n_heads, seq_len, seq_len).cuda()

        # Apply mask
        attention_scores = torch.einsum('bqhd,bkhd->bhqk', Q, K) / self.scale
        attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # Apply softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Apply attention
        output = torch.einsum('bhqk,bkhd->bqhd', attention_probs, V).reshape(batch_size, seq_len, self.d_model)

        return output


class MultiHeadedAttention(nn.Module):
    def __init__(self, model_dimension, number_of_heads, dropout_probability):
        super().__init__()
        assert model_dimension % number_of_heads == 0, f'Model dimension must be divisible by the number of heads.'

        self.head_dimension = int(model_dimension / number_of_heads)
        self.number_of_heads = number_of_heads

        self.qkv_nets = get_clones(nn.Linear(model_dimension, model_dimension), 3)  # identity activation hence "nets"
        self.out_projection_net = nn.Linear(model_dimension, model_dimension)

        self.attention_dropout = nn.Dropout(p=dropout_probability)  # no pun intended, not explicitly mentioned in paper
        self.softmax = nn.Softmax(dim=-1)  # -1 stands for apply the softmax along the last dimension

        # self.attention_weights = None  # for visualization purposes, I cache the weights here (translation_script.py)

    def attention(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dimension)
        attention_weights = self.softmax(scores)
        attention_weights = self.attention_dropout(attention_weights)
        intermediate_token_representations = torch.matmul(attention_weights, value)

        return intermediate_token_representations, attention_weights  # attention weights for visualization purposes

    def forward(self, x):
        batch_size = x.shape[0]
        query, key, value = [net(x).view(batch_size, -1, self.number_of_heads, self.head_dimension).transpose(1, 2)
                             for net, x in zip(self.qkv_nets, (x, x, x))]
        intermediate_token_representations, attention_weights = self.attention(query, key, value)

        reshaped = intermediate_token_representations.transpose(1, 2).reshape(batch_size, -1,
                                                                              self.number_of_heads * self.head_dimension)
        # forward
        token_representations = self.out_projection_net(reshaped)
        return token_representations


def get_clones(module, num_of_deep_copies):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_of_deep_copies)])


class ModalityTransformer(nn.Module):
    def __init__(self, model_dimension, number_of_heads, attention_type, dropout_probability):
        super().__init__()
        encoder_layer = EncoderLayer(model_dimension, number_of_heads, attention_type, dropout_probability)
        self.encoder = Encoder(encoder_layer)
        self.init_params()

    def init_params(self, default_initialization=False):
        if not default_initialization:
            # model.named_parameters
            for name, p in self.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, input):
        src_representations_batch1 = self.encoder(input)
        return src_representations_batch1


class Encoder(nn.Module):
    def __init__(self, encoder_layer):
        super().__init__()
        assert isinstance(encoder_layer, EncoderLayer), f'Expected EncoderLayer got {type(encoder_layer)}.'
        self.encoder_layer = encoder_layer
        self.ffn_layer = PoswiseFeedForwardNet(encoder_layer.model_dimension, encoder_layer.model_dimension * 2)

    def forward(self, x):
        src_representations_batch = self.encoder_layer(x)
        representations = self.ffn_layer(src_representations_batch)
        return representations


class EncoderLayer(nn.Module):
    def __init__(self, model_dimension, number_of_heads, attention_type, dropout_probability):
        super().__init__()
        if attention_type == 'causal':
            self.sublayer = SublayerLogic(model_dimension, dropout_probability)
            self.mha = CausalAttention(model_dimension=model_dimension, n_heads=number_of_heads,
                                             dropout_probability=dropout_probability)
        else:
            self.sublayer = SublayerLogic(model_dimension, dropout_probability)
            self.mha = MultiHeadedAttention(model_dimension=model_dimension, number_of_heads=number_of_heads,
                                       dropout_probability=dropout_probability)
        self.model_dimension = model_dimension
        self.norm = nn.LayerNorm(model_dimension)

    def forward(self, x):
        # 多头注意
        encoder_self_attention = lambda x: self.mha(x)
        src_representations_batch = self.norm(self.sublayer(x, encoder_self_attention))
        return src_representations_batch


class SublayerLogic(nn.Module):
    def __init__(self, model_dimension, dropout_probability):
        super().__init__()
        self.norm = nn.LayerNorm(model_dimension)
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, x, mha):
        return x + self.dropout(mha(self.norm(x)))
