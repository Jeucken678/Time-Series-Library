import torch
from data_provider.data_loader import PriceImputationDataset  # 导入你的数据集类

# 1. 配置数据集路径（改为你实际的 price_imputation_datasets 文件夹路径）
root_path =  "/home/zenghuizhu/Time-Series-Library/price/price_imputation_datasets"
flag = "train"  # 可测试 train/val/test 任意数据集

# 2. 加载数据集（触发 __init__ 和 __getitem__ 方法）
dataset = PriceImputationDataset(root_path, flag=flag)

# 3. 遍历前3个样本，触发掩码验证打印
print(f"数据集总样本数：{len(dataset)}")
for idx in range(3):
    # 调用 __getitem__ 方法，触发之前加的打印逻辑
    sample = dataset[idx]
    print("-" * 50)  # 分隔不同样本的输出