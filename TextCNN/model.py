import torch
import torch.nn as nn
import torch.nn.functional as F


# 构建模型
class TextCNN(nn.Module):
    def __init__(self, vocab_size, config):
        super(TextCNN, self).__init__()
        self.kernel_sizes = config.kernel_sizes
        self.hidden_dim = config.embed_dim
        self.num_channel = config.num_channel
        self.num_class = config.num_class
        self.word_embedding = nn.Embedding(vocab_size, config.embed_dim)  # 词向量，这里直接随机

        self.convs = nn.ModuleList(
            [nn.Conv2d(self.num_channel, config.num_kernel, (kernel, config.embed_dim))
             for kernel in self.kernel_sizes])  # 卷积层

        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_kernel * 3, self.num_class)  # 全连接层

    def forward(self, x):
        x = self.word_embedding(x)  # 第一层
        x = x.permute(1, 0, 2).unsqueeze(1)
        x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]  # 第二层conv，每三个提取一个特征
        x = [F.max_pool1d(h, h.size(2)).squeeze(2) for h in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)  # 随机抛弃某些向量，0.5左右的概率
        logits = self.fc(x)
        return logits
