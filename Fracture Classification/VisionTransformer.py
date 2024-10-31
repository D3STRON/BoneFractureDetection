import os
import pandas as pd
import torch
import torch.nn
import pandas as pd 
from sklearn.model_selection import train_test_split
import torch.nn as nn
from tqdm import tqdm, trange
import math
import numpy as np


def get_positional_embeddings(batch_size, sequence_length, d, device):
    # Create a position tensor [sequence_length, 1]
    position = torch.arange(sequence_length).unsqueeze(1).float()
    # Create a divisor tensor for sine and cosine [1, d // 2] based on the formula
    div_term = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    
    # Initialize embeddings with batch and sequence dimensions
    pos_embedding = torch.zeros(batch_size, sequence_length, d)
    
    # Apply sine to even indices and cosine to odd indices
    pos_embedding[:, :, 0::2] = torch.sin(position * div_term)  # even indices
    pos_embedding[:, :, 1::2] = torch.cos(position * div_term)  # odd indices
    
    return pos_embedding.to(device)

class PatchEncoder(nn.Module):
    def __init__(self, config):
        super(PatchEncoder, self).__init__()
        self.encoding_size = config['encoding_size']
        self.img_dim = config['img_dim']
        self.dropout = config['dropout']
        self.patch_size = config['patch_size']
        assert self.img_dim % self.patch_size == 0, ("Image is not compatible with number of patches")
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.encoding_size))
        self.n_patches = (self.img_dim//self.patch_size)**2
        self.encoder = nn.Conv2d(config['channels'], self.encoding_size, kernel_size=self.patch_size, stride=self.patch_size)
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, x):
        out = self.encoder(x)
        cls_token = self.cls_token.expand((x.shape[0], -1, -1))
        out = out.flatten(-2).transpose(-1, -2)
        out = torch.cat((cls_token, out), dim = 1)
#         print(out.shape, self.pos_encoder.shape)
        pos_enc = get_positional_embeddings(out.shape[0], out.shape[1], out.shape[2], device=x.device)
        out = out + pos_enc 
        return self.dropout_layer(out)
        



class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        

        self.d_model = config['encoding_size']           # Total dimension of the model
        self.num_heads = config['num_heads']       # Number of attention heads
        assert self.d_model % self.num_heads == 0, 'd_model must be divisible by num_heads'
        self.d_k = self.d_model // self.num_heads  # Dimnsion of each head. We assume d_v = d_k

        # Linear transformations for queries, keys, and values
        self.W_q = nn.Linear(self.d_model, self.d_model)
        self.W_k = nn.Linear(self.d_model, self.d_model)
        self.W_v = nn.Linear(self.d_model, self.d_model)

        # Final linear layer to project the concatenated heads' outputs back to d_model dimensions
        self.W_o = nn.Linear(self.d_model, self.d_model)

    def scaled_dot_product_attention(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-2, -1))
        attn_weights = torch.softmax(scores / math.sqrt(self.d_k), dim = -1)
        output = torch.matmul(attn_weights, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, num_heads, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, x):
        Q_proj = self.W_q(x)
        K_proj = self.W_k(x)
        V_proj = self.W_v(x)

        Q_proj_split = self.split_heads(Q_proj)
        K_proj_split = self.split_heads(K_proj)
        V_proj_split = self.split_heads(V_proj)

        attention_scores = self.scaled_dot_product_attention(Q_proj_split, K_proj_split, V_proj_split)

        output = self.combine_heads(attention_scores)
        output = self.W_o(output)

        return output

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoding_size = config['encoding_size']
        self.intermediate_size = config['intermediate_size']
        self.dense_1 = nn.Linear(self.encoding_size, self.intermediate_size)
        self.activation = nn.GELU()
        self.dense_2 = nn.Linear(self.intermediate_size, self.encoding_size)
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x
    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoding_size = config['encoding_size']
        self.attention = MultiHeadAttention(config)
        self.layernorm_1 = nn.LayerNorm(self.encoding_size)
        self.mlp = MLP(config)
        self.layernorm_2 = nn.LayerNorm(self.encoding_size)

    def forward(self, x):
        attention_output = self.attention(self.layernorm_1(x))
        x = x + attention_output
        mlp_output = self.mlp(self.layernorm_2(x))
        x = x + mlp_output
        return x
    
class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(config["num_hidden_layers"]):
            block = Block(config)
            self.blocks.append(block)

    def forward(self, x):
        all_attentions = []
        for block in self.blocks:
            x = block(x)
        return x
    
class ViTForClassfication(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoding_size = config["encoding_size"]
        self.num_classes = config["num_classes"]
        self.embedding = PatchEncoder(config)
        self.encoder = Encoder(config)
        self.classifier = nn.Linear(self.encoding_size, self.num_classes)

    def forward(self, x):
        embedding_output = self.embedding(x)
        encoder_output = self.encoder(embedding_output)
        logits = self.classifier(encoder_output[:, 0])
        return logits
