from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Imputation(Exp_Basic):
    def __init__(self, args):
        super(Exp_Imputation, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].TimesNet(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        # 假设data_provider已修改为返回包含nan_mask的6元素元组
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            # 接收6个值（新增nan_mask）
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, target_mask, nan_mask) in enumerate(vali_loader):
                # 数据类型转换并移动到设备
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                target_mask = target_mask.float().to(self.device)  # 需要补全的位置（1表示需要补全）
                nan_mask = nan_mask.float().to(self.device)  # 原始数据有效性（0表示有效，1表示无效）

                # 生成有效计算掩码：原始数据有效（nan_mask=0）且需要补全（target_mask=1）
                valid_mask = (nan_mask == 0.0) & (target_mask == 1.0)

                # 掩盖输入中需要补全的位置（与训练逻辑一致）
                inp = batch_x.masked_fill(target_mask == 1.0, 0.0)  # 仅掩盖需要补全的位置

                # 模型前向传播
                outputs = self.model(inp, batch_x_mark, None, None, target_mask)

                # 处理特征维度
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]
                # 调整掩码维度以匹配特征维度
                valid_mask = valid_mask[:, :, f_dim:] if f_dim != -1 else valid_mask  # 关键：对齐特征维度

                # 仅在有效计算掩码位置计算损失
                if valid_mask.any():  # 避免掩码全为False导致的错误
                    loss = criterion(outputs[valid_mask], batch_y[valid_mask])
                    total_loss.append(loss.item())

        # 计算平均损失（处理空掩码情况）
        total_loss = np.average(total_loss) if total_loss else 0.0
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            # 接收6个值（新增nan_mask）
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, target_mask, nan_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                # 数据类型转换
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                target_mask = target_mask.float().to(self.device)
                nan_mask = nan_mask.float().to(self.device)

                # 生成有效掩码
                valid_mask = (nan_mask == 0.0) & (target_mask == 1.0)

                # 关键修改：跳过无有效补全位置的batch
                if not valid_mask.any():
                    print(f"跳过无有效补全位置的batch {i}")
                    continue  # 不更新参数，直接进入下一个batch

                # 后续原有逻辑（掩盖输入、模型前向传播、损失计算等）
                inp = batch_x.masked_fill(target_mask == 1.0, 0.0)
                outputs = self.model(inp, batch_x_mark, None, None, target_mask)

                # 处理特征维度
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                batch_x = batch_x[:, :, f_dim:]  # 用batch_x作为真实值（原始有效数据）
                # 调整掩码维度以匹配特征维度
                valid_mask = valid_mask[:, :, f_dim:] if f_dim != -1 else valid_mask

                # 仅在有效计算掩码位置计算损失
                if valid_mask.any():  # 避免掩码全为False导致的错误
                    loss = criterion(outputs[valid_mask], batch_x[valid_mask])
                    train_loss.append(loss.item())
                else:
                    loss = torch.tensor(0.0, device=self.device)  # 无有效位置时损失为0

                # 打印迭代信息
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # 反向传播与参数更新
                loss.backward()
                model_optim.step()

            # 打印epoch信息
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss) if train_loss else 0.0
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        self.model.eval()

        preds = []
        trues = []

        with torch.no_grad():
            # 接收6个值（新增nan_mask）
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, target_mask, nan_mask) in enumerate(test_loader):
                # 数据类型转换
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                target_mask = target_mask.float().to(self.device)
                nan_mask = nan_mask.float().to(self.device)

                # 生成有效计算掩码
                valid_mask = (nan_mask == 0.0) & (target_mask == 1.0)

                # 掩盖输入
                inp = batch_x.masked_fill(target_mask == 1.0, 0.0)

                # 模型预测
                outputs = self.model(inp, batch_x_mark, None, None, target_mask)

                # 处理特征维度
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]
                # 调整掩码维度
                valid_mask = valid_mask[:, :, f_dim:] if f_dim != -1 else valid_mask

                # 仅保留有效计算掩码位置的结果
                if valid_mask.any():
                    outputs_masked = outputs[valid_mask]
                    batch_y_masked = batch_y[valid_mask]
                    preds.append(outputs_masked.cpu().numpy())
                    trues.append(batch_y_masked.cpu().numpy())

        # 拼接结果并计算指标
        if preds and trues:
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            print('test shape:', preds.shape, trues.shape)

            mse = np.mean((preds - trues) **2)
            mae = np.mean(np.abs(preds - trues))
            print(f'Test MSE: {mse:.6f}, Test MAE: {mae:.6f}')
        else:
            mse, mae = 0.0, 0.0
            print("No valid test samples to evaluate.")

        # 保存结果
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save(folder_path + 'preds.npy', preds)
        np.save(folder_path + 'trues.npy', trues)

        return mse, mae