import torch
import torch.nn as nn
import math
import pytorch_lightning as pl
from torch.optim import Adam
from utils import get_disk_mask

class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim, q_dim, kv_dim, num_heads):
        super(CrossAttentionLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.scale = self.num_heads ** -0.5

        self.query_proj = nn.Linear(q_dim, embed_dim)
        self.key_proj = nn.Linear(kv_dim, embed_dim)
        self.value_proj = nn.Linear(kv_dim, embed_dim)

        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        """
        :param query: (batch_size, query_len, embed_dim)
        :param key: (batch_size, key_len, embed_dim)
        :param value: (batch_size, value_len, embed_dim)
        :param mask: (batch_size, query_len, key_len) 可选的 mask，防止某些位置被注意到
        :return: output, attention_weights
        """
        B, N1, _ = query.shape
        _, N2, _ = key.shape

        Q = self.query_proj(query).reshape(B, N1, self.num_heads, int(self.embed_dim / self.num_heads)).permute(0, 2, 1, 3)  # (batch, num_heads, N, dim)
        K = self.key_proj(key).reshape(B, N2, self.num_heads, int(self.embed_dim / self.num_heads)).permute(0, 2, 1, 3)  # (batch, num_heads, N, dim)
        V = self.value_proj(value).reshape(B, N2, self.num_heads, int(self.embed_dim / self.num_heads)).permute(0, 2, 1, 3)  # (batch, num_heads, N, dim)

        att = (Q @ K.transpose(-2, -1)) * self.scale  # (batch, num_heads, N, N)
        att = att.softmax(dim=-1)
        attention_output = (att @ V).transpose(1, 2).flatten(2)  # B,N,dim

        output_ = self.output_proj(attention_output)  # (batch_size, query_len, embed_dim)

        return output_

class CrossAttentionModel(nn.Module):
    def __init__(self, embed_dim, his_dim, ge_dim, num_heads=8):
        super(CrossAttentionModel, self).__init__()
        self.cross_attention1 = CrossAttentionLayer(embed_dim=embed_dim, q_dim=his_dim, kv_dim=ge_dim, num_heads=num_heads)
        self.cross_attention2 = CrossAttentionLayer(embed_dim=embed_dim, q_dim=ge_dim, kv_dim=his_dim, num_heads=num_heads)
        self.cross_attention3 = CrossAttentionLayer(embed_dim=embed_dim, q_dim=embed_dim, kv_dim=embed_dim, num_heads=num_heads)

    def forward(self, A, B):

        C = self.cross_attention1(A, B, B)

        D = self.cross_attention2(B, A, A)

        final_output = self.cross_attention3(C, D, D)

        return final_output

class Linear(nn.Module):
    def __init__(self, num_input, num_output, alpha=0.01, beta=0.01, bias=False, func=True):
        super(Linear, self).__init__()
        self.weights = nn.Parameter(torch.FloatTensor(num_input,num_output))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(num_output))
        else:
            self.register_parameter('bias', None)
        if func:
            self.func = nn.ELU(alpha=alpha, inplace=True)
        else:
            self.func = nn.Identity()
        self.beta = beta
        self.reset_parameters()

    def reset_parameters(self):
        std = 1./math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, x, indices=None):

        if indices is None:
            output = torch.matmul(x, self.weights)
            if self.bias is not None:
                output = output + self.bias
        else:
            weight = self.weights[:, indices]
            output = torch.matmul(x, weight)
            if self.bias is not None:
                output = output + self.bias[indices]
        output = self.func(output) + self.beta
        return output

class SpaPSC(pl.LightningModule):
    def __init__(self, lr, num_features, num_genes, num_embeddings, radius, bias=False):
        super(SpaPSC, self).__init__()

        self.lr = lr
        self.radius = radius
        self.num_embeddings = num_embeddings

        self.MCA1 = CrossAttentionModel(embed_dim=num_embeddings, his_dim=num_features, ge_dim=num_genes)
        self.l1 = Linear(num_embeddings, num_embeddings, bias=bias)
        self.MCA2 = CrossAttentionModel(embed_dim=num_embeddings, his_dim=num_embeddings, ge_dim=num_genes)
        self.l2 = Linear(num_embeddings, num_embeddings, bias=bias)

        self.output_module = nn.Sequential(Linear(num_embeddings, 1024, bias=bias),  # 注意调整alpha和beta
                                           Linear(1024, 512, bias=bias),
                                           Linear(512, 512, bias=bias),
                                           Linear(512, 512, bias=bias))
        self.fal_out = Linear(512, num_genes, bias=bias)
        self.save_hyperparameters()

    def get_Multi_Fea(self, his_fea, gene_fea):
        x = self.MCA1.forward(his_fea, gene_fea)
        x = self.l1(x) + x
        x = self.MCA2.forward(x, gene_fea)
        x = self.l2(x) + x

        return x

    def get_gene(self, x, indices=None):
        x = self.output_module.forward(x)
        x = self.fal_out(x, indices)
        return x

    def forward(self, x, indices=None):
        his_fea = x['his'].to(torch.float32)
        gene_fea = x['gene']
        x = self.get_Multi_Fea(his_fea, gene_fea)
        x = self.get_gene(x, indices)
        return x

    def training_step(self, batch, batch_idx):
        x, y_mean = batch
        y_pred = self.forward(x)
        mask = get_disk_mask(55/16)
        mask = torch.BoolTensor(mask).to('cuda')
        y_pred = y_pred.reshape(y_pred.shape[0], mask.shape[0], mask.shape[1], y_pred.shape[2])
        y_pred = torch.masked_select(y_pred, mask.unsqueeze(0).unsqueeze(-1)).view(y_pred.shape[0], -1, y_pred.shape[-1])

        y_mean_pred = y_pred.mean(-2)

        mse = ((y_mean_pred - y_mean)**2).mean()
        loss = mse
        self.log('loss', loss**0.5, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer
