# -*- coding: utf-8 -*-
import os
import torch
from config import parse_config
from data_loader import DataBatchIterator
from data_loader import PAD
from model import TextCNN
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import logging


def build_textcnn_model(vocab, config, train=True):
    model = TextCNN(vocab.vocab_size, config)
    if train:
        model.train()
    else:
        model.eval()

    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()
    return model


def train_textcnn_model(model, train_data, valid_data, padding_idx, config):
    # Build optimizer.
    # embed不可以训练，他是随机生成的
    # params = [p for k, p in model.named_parameters(
    # ) if p.requires_grad and "embed" not in k]
    # 设置可以被训练的参数
    # embed参数训练
    params = [p for k, p in model.named_parameters() if p.requires_grad]
    # 优化器adam
    optimizer = Adam(params, lr=config.lr)
    # 对n个样本的loss求和作为得分
    criterion = CrossEntropyLoss(reduction="sum")
    # 训练（调整权重）
    model.train()

    for epoch in range(1, config.epochs + 1):
        train_data_iter = iter(train_data)
        # 抱一个包装器，强制转换
        for idx, batch in enumerate(train_data_iter):
            # 遍历取batch
            model.zero_grad()
            # 模型的参数梯度归零
            ground_truth = batch.label
            # 目标结果
            # batch_first = False
            outputs = model(batch.sent)
            loss = criterion(outputs, ground_truth)  #计算损失
            loss.backward()  #反向传播
            optimizer.step()  # 更新参数

            if idx % 20 == 0:
                valid_loss = valid_textcnn_model(
                    model, valid_data, criterion, config)
                # 处理
                print("epoch {0:d} [{1:d}/{2:d}], valid loss: {3:.2f}".format(
                    epoch, idx, train_data.num_batches, valid_loss))
                model.train()


def valid_textcnn_model(model, valid_data, criterion, config):
    # Build optimizer.
    # params = [p for k, p in model.named_parameters(
    # ) if p.requires_grad and "embed" not in k]
    model.eval()
    total_loss = 0
    valid_data_iter = iter(valid_data)
    for idx, batch in enumerate(valid_data_iter):
        model.zero_grad()
        ground_truth = batch.label
        # batch_first = False
        outputs = model(batch.sent)
        # probs = model.generator(decoder_outputs)
        loss = criterion(outputs, batch.label) 
        # loss 打印
        # 处理
        total_loss += loss
        # break
    return total_loss


def main():
    # 读配置文件
    config = parse_config()
    # 载入训练集合
    train_data = DataBatchIterator(
        config=config,
        is_train=True,
        dataset="train",
        batch_size=config.batch_size,
        shuffle=True)
    train_data.load()

    vocab = train_data.vocab

    # 载入测试集合
    valid_data = DataBatchIterator(
        config=config,
        is_train=False,
        dataset="dev",
        batch_size=config.batch_size)
    valid_data.set_vocab(vocab)
    valid_data.load()

    # 构建textcnn模型
    model = build_textcnn_model(vocab, config, train=True)

    print(model)

    # Do training.
    padding_idx = vocab.stoi[PAD]
    train_textcnn_model(model, train_data,
                        valid_data, padding_idx, config)
    torch.save(model, '%s.pt' % (config.save_model))

    # 测试时
    # checkpoint = torch.load(config.save_model+".pt",
    #                      map_location = config.device)
    # checkpoint
    # model = build_textcnn_model(
    #     vocab, config, train=True)
    # .....


if __name__ == "__main__":
    main()
