import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.metrics import FFT_for_Period  # 复用库中FFT工具函数


class TimesBlock(nn.Module):
    def __init__(self, args):
        super(TimesBlock, self).__init__()
        self.args = args
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.top_k = args.top_k
        self.d_model = args.d_model

        # 2D卷积块（遵循库中Inception结构设计）
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=self.d_model,
                out_channels=self.d_model * 2,
                kernel_size=(3, 1),
                padding=(1, 0)
            ),
            nn.GELU(),
            nn.Conv2d(
                in_channels=self.d_model * 2,
                out_channels=self.d_model,
                kernel_size=(3, 1),
                padding=(1, 0)
            ),
            nn.GELU()
        )

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        # FFT周期检测（复用库中工具函数）
        period_list, period_weight = FFT_for_Period(x, self.top_k)

        # 多周期特征提取
        res = []
        for i in range(self.top_k):
            period = period_list[i]
            # 序列补齐（确保能被周期整除）
            pad_len = (period - seq_len % period) % period
            x_padded = F.pad(x, (0, 0, 0, pad_len))  # (batch_size, seq_len+pad_len, d_model)
            # 1D -> 2D: (batch_size, num_blocks, period, d_model)
            num_blocks = (seq_len + pad_len) // period
            x_2d = x_padded.reshape(batch_size, num_blocks, period, d_model).permute(0, 3, 1, 2)
            # 2D卷积
            x_2d = self.conv(x_2d)  # (batch_size, d_model, num_blocks, period)
            # 2D -> 1D
            x_1d = x_2d.permute(0, 2, 3, 1).reshape(batch_size, -1, d_model)
            res.append(x_1d[:, :seq_len, :])  # 截断至原始长度

        # 周期权重聚合
        res = torch.stack(res, dim=-1)  # (batch_size, seq_len, d_model, top_k)
        period_weight = F.softmax(period_weight, dim=1).unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, top_k)
        return torch.sum(res * period_weight, dim=-1)  # (batch_size, seq_len, d_model)


class TimesNet(nn.Module):
    def __init__(self, args):
        super(TimesNet, self).__init__()
        self.args = args
        self.task_name = args.task_name
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.d_model = args.d_model
        self.input_size = args.input_size

        # 输入嵌入（映射原始特征到d_model）
        self.embedding = nn.Linear(self.input_size, self.d_model)
        # 堆叠TimesBlock
        self.encoder = nn.ModuleList([
            TimesBlock(args) for _ in range(args.e_layers)
        ])
        # 输出投影（根据任务类型调整）
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.projection = nn.Linear(self.d_model, self.input_size)
        elif self.task_name == 'imputation':
            self.projection = nn.Linear(self.d_model, self.input_size)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        适配库中多任务接口：
        - 预测任务：x_enc为历史序列 (batch_size, seq_len, input_size)
        - 插补任务：x_enc为含缺失的序列，mask为缺失掩码
        """
        # 输入嵌入
        x = self.embedding(x_enc)  # (batch_size, seq_len, d_model)

        # 编码器处理
        for block in self.encoder:
            x = block(x)

        # 任务适配
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            # 预测任务：取最后pred_len长度
            x = x[:, -self.pred_len:, :]
        # 插补任务：直接输出完整序列

        # 投影到原始特征维度
        return self.projection(x)