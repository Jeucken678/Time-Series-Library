import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, args):
        super(TimesBlock, self).__init__()
        self.args = args
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.top_k = args.top_k
        self.d_model = args.d_model
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
        batch_size, seq_len, d_model = x.shape
        period_list, period_weight = FFT_for_Period(x, self.top_k)
        res = []
        for i in range(self.top_k):
            period = period_list[i]
            pad_len = (period - seq_len % period) % period
            x_padded = F.pad(x, (0, 0, 0, pad_len))
            num_blocks = (seq_len + pad_len) // period
            x_2d = x_padded.reshape(batch_size, num_blocks, period, d_model).permute(0, 3, 1, 2)
            x_2d = self.conv(x_2d)
            x_1d = x_2d.permute(0, 2, 3, 1).reshape(batch_size, -1, d_model)
            res.append(x_1d[:, :seq_len, :])
        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1).unsqueeze(1).unsqueeze(1)
        return torch.sum(res * period_weight, dim=-1)


class TimesNet(nn.Module):
    def __init__(self, args):
        super(TimesNet, self).__init__()
        self.args = args
        self.task_name = args.task_name
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.d_model = args.d_model
        self.input_size = args.input_size
        self.embedding = nn.Linear(self.input_size, self.d_model)
        self.encoder = nn.ModuleList([
            TimesBlock(args) for _ in range(args.e_layers)
        ])
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.projection = nn.Linear(self.d_model, self.input_size)
        elif self.task_name == 'imputation':
            self.projection = nn.Linear(self.d_model, self.input_size)

        # 新增：归一化用的epsilon（避免除零）
        self.eps = 1e-5

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        关键修改：添加基于mask的输入归一化（仅对插补任务生效）
        x_enc: (batch_size, seq_len, input_size) —— 模型输入（已处理NaN和遮盖）
        mask: (batch_size, seq_len, input_size) —— target_mask（1=需要补全，0=有效数据）
        """
        batch_size, seq_len, input_size = x_enc.shape

        # ========== 新增：插补任务专属——基于mask的归一化 ==========
        if self.task_name == 'imputation' and mask is not None:
            # 1. 标记有效数据位置（mask=0是有效数据，mask=1是需要补全的位置）
            valid_mask = (mask == 0.0).float()  # (batch_size, seq_len, input_size)

            # 2. 计算有效数据的均值和方差（按每个样本、每个特征单独计算）
            valid_count = valid_mask.sum(dim=1, keepdim=True)  # 每个样本-特征的有效数据量
            valid_count = torch.clamp(valid_count, min=self.eps)  # 避免除零

            # 均值：有效数据的平均
            mean = (x_enc * valid_mask).sum(dim=1, keepdim=True) / valid_count
            # 方差：有效数据的方差（加eps避免方差为0）
            var = ((x_enc - mean) * valid_mask).pow(2).sum(dim=1, keepdim=True) / valid_count + self.eps
            std = torch.sqrt(var)

            # 3. 对输入归一化（仅用有效数据的统计特性）
            x_enc = (x_enc - mean) / std
        # ========================================================

        # 输入嵌入（归一化后的数据）
        x = self.embedding(x_enc)  # (batch_size, seq_len, d_model)

        # 编码器处理
        for block in self.encoder:
            x = block(x)

        # 任务适配
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            x = x[:, -self.pred_len:, :]

        # 投影到原始特征维度
        output = self.projection(x)  # (batch_size, seq_len, input_size)

        # ========== 新增：反归一化（恢复原始数据尺度） ==========
        if self.task_name == 'imputation' and mask is not None:
            output = output * std + mean  # 反归一化，和真实值尺度一致
        # ========================================================

        return output