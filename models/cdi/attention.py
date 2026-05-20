import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        b, c = x.shape  # b: batch_size, c: embed_size
        query = self.query(x).view(b, 1, c)  # bs x 1 x dim
        key = self.key(x).view(b, 1, c)  # bs x 1 x dim
        value = self.value(x).view(b, 1, c)  # bs x 1 x dim

        scores = torch.bmm(query, key.transpose(1, 2)) / torch.sqrt(torch.tensor(c, dtype=torch.float32))  # bs x 1 x 1
        attention_weights = F.softmax(scores, dim=-1)  # bs x 1 x 1
        output = torch.bmm(attention_weights, value).squeeze(1)  # bs x dim
        return output
