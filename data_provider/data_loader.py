import os
import torch
from torch.utils.data import Dataset


class PriceImputationDataset(Dataset):
    """适配 price_imputation_datasets 文件夹的缺失值补全数据集"""
    def __init__(self, root_path, flag="train"):
        # 拼接文件路径（与你的数据集文件名完全匹配）
        self.data = torch.load(os.path.join(root_path, f"{flag}_data.pt"))
        self.nan_mask = torch.load(os.path.join(root_path, f"{flag}_nan_mask.pt"))
        self.target_mask = torch.load(os.path.join(root_path, f"{flag}_target_mask.pt"))
        # 遮盖预测目标位置（避免信息泄露）
        self.input_data = self.data * (1 - self.target_mask)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "x": self.input_data[idx],   # 模型输入（已遮盖预测目标）
            "y_true": self.data[idx],    # 真实值（含缺失）
            "nan_mask": self.nan_mask[idx],
            "target_mask": self.target_mask[idx]
        }


class PriceForecastDataset(Dataset):
    """适配 pridict_price_datasets 文件夹的价格预测数据集"""
    def __init__(self, root_path, flag="train"):
        # 拼接文件路径（与你的数据集文件名完全匹配）
        self.data_x = torch.load(os.path.join(root_path, f"{flag}_data_x.pt"))  # 历史序列
        self.data_y = torch.load(os.path.join(root_path, f"{flag}_data_y.pt"))  # 未来序列
        self.x_nan_mask = torch.load(os.path.join(root_path, f"{flag}_x_nan_mask.pt"))
        self.y_target_mask = torch.load(os.path.join(root_path, f"{flag}_y_target_mask.pt"))

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        return {
            "x": self.data_x[idx],       # 历史序列（模型输入）
            "y_true": self.data_y[idx],  # 未来序列真实值
            "x_nan_mask": self.x_nan_mask[idx],
            "y_target_mask": self.y_target_mask[idx]
        }