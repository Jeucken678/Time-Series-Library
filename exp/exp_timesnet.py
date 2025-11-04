from exp.exp_basic import Exp_Basic
from models import TimesNet
from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

import warnings
warnings.filterwarnings('ignore')


class Exp_TimesNet(Exp_Basic):
    def __init__(self, args):
        super(Exp_TimesNet, self).__init__(args)

    def _build_model(self):
        model = TimesNet(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args
        data_set, data_loader = None, None
        if flag == 'train':
            dataset_name = args.data
            if dataset_name == 'ETTh1':
                data_set = Dataset_ETT_hour(
                    root_path=args.root_path,
                    data_path=args.data_path,
                    flag=flag,
                    size=[args.seq_len, args.label_len, args.pred_len],
                    features=args.features,
                    target=args.target,
                    inverse=args.inverse
                )
            elif dataset_name == 'custom':
                data_set = Dataset_Custom(
                    root_path=args.root_path,
                    data_path=args.data_path,
                    flag=flag,
                    size=[args.seq_len, args.label_len, args.pred_len],
                    features=args.features,
                    target=args.target,
                    inverse=args.inverse
                )
            # 其他数据集类型...

            data_loader = DataLoader(
                data_set,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                drop_last=True
            )
        else:
            # 验证/测试集加载（逻辑类似，shuffle=False）
            dataset_name = args.data
            if dataset_name == 'ETTh1':
                data_set = Dataset_ETT_hour(
                    root_path=args.root_path,
                    data_path=args.data_path,
                    flag=flag,
                    size=[args.seq_len, args.label_len, args.pred_len],
                    features=args.features,
                    target=args.target,
                    inverse=args.inverse
                )
            # 其他数据集类型...

            data_loader = DataLoader(
                data_set,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers
            )
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.task_name == 'imputation':
            return nn.MSELoss()  # 插补任务用MSE
        else:
            return nn.MSELoss()  # 预测任务用MSE

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # 前向传播（适配多任务）
                if self.args.task_name in ['long_term_forecast', 'short_term_forecast']:
                    outputs = self.model(batch_x, batch_x_mark, None, None)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                elif self.args.task_name == 'imputation':
                    outputs = self.model(batch_x, batch_x_mark, None, None, mask=batch_x_mark)  # 假设mask用x_mark传递
                    batch_y = batch_x  # 插补任务目标是输入本身（非缺失部分）

                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * len(train_loader) - i)
                    print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print(f"Epoch: {epoch + 1} | Train Loss: {np.mean(train_loss):.7f}")
            vali_loss = self.vali(vali_loader, criterion)
            test_loss = self.vali(test_loader, criterion)

            print(f"Epoch: {epoch + 1} | Vali Loss: {vali_loss:.7f} | Test Loss: {test_loss:.7f}")
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def vali(self, data_loader, criterion):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if self.args.task_name in ['long_term_forecast', 'short_term_forecast']:
                    outputs = self.model(batch_x, batch_x_mark, None, None)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                elif self.args.task_name == 'imputation':
                    outputs = self.model(batch_x, batch_x_mark, None, None)
                    batch_y = batch_x

                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())
        self.model.train()
        return np.mean(total_loss)

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        self.model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if self.args.task_name in ['long_term_forecast', 'short_term_forecast']:
                    outputs = self.model(batch_x, batch_x_mark, None, None)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                elif self.args.task_name == 'imputation':
                    outputs = self.model(batch_x, batch_x_mark, None, None)
                    batch_y = batch_x

                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        # 计算评估指标（复用库中metric函数）
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(f"mae:{mae}, mse:{mse}, rmse:{rmse}, mape:{mape}, mspe:{mspe}")

        return {
            'mae': mae, 'mse': mse, 'rmse': rmse,
            'mape': mape, 'mspe': mspe
        }

    # exp/exp_timesnet.py
    class Exp_TimesNet(Exp_Basic):
        def _get_data(self, flag):
            args = self.args
            if args.task_name == 'imputation':
                # 缺失值补全任务
                dataset = PriceImputationDataset(
                    root_path=args.root_path,
                    flag=flag
                )
            elif args.task_name == 'forecast':
                # 价格预测任务
                dataset = PriceForecastDataset(
                    root_path=args.root_path,
                    flag=flag
                )
            else:
                raise ValueError(f"Unknown task: {args.task_name}")

            data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=(flag == 'train'),
                num_workers=args.num_workers,
                drop_last=(flag == 'train')
            )
            return dataset, data_loader