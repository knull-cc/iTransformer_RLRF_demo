from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
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


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.teacher = None

    # ============== Distill helpers ==============
    def _build_teacher_model(self):
        if not getattr(self.args, 'enable_distill', False):
            return None
        # select teacher architecture
        teacher_model_name = getattr(self.args, 'teacher_model', self.args.model)
        teacher_args = self.args
        teacher = self.model_dict[teacher_model_name].Model(teacher_args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            teacher = nn.DataParallel(teacher, device_ids=self.args.device_ids)
        teacher = teacher.to(self.device)
        # load checkpoint if provided
        ckpt = getattr(self.args, 'teacher_ckpt', '')
        if ckpt and os.path.exists(ckpt):
            state = torch.load(ckpt, map_location=self.device)
            if 'state_dict' in state:
                state = state['state_dict']
            # try both stripping and adding 'module.' prefix
            def _load(sd):
                try:
                    teacher.load_state_dict(sd, strict=False)
                    return True
                except Exception:
                    return False
            if not _load(state):
                # strip 'module.'
                fixed = {k.replace('module.', ''): v for k, v in state.items()}
                if not _load(fixed):
                    # add 'module.'
                    fixed2 = {('module.' + k if not k.startswith('module.') else k): v for k, v in state.items()}
                    _load(fixed2)
        # freeze
        for p in teacher.parameters():
            p.requires_grad = False
        teacher.eval()
        return teacher

    def _get_horizon_weights(self, pred_len, device):
        if not getattr(self.args, 'horizon_weighting', True):
            return torch.ones(pred_len, device=device)
        # linear ramp: later steps heavier
        w = torch.linspace(1.0, 2.0, steps=pred_len, device=device)
        # normalize to keep scale stable
        return w / w.mean()

    def _weighted_mse(self, pred, target, weights):
        # pred/target: [B, L, D], weights: [L]
        se = (pred - target) ** 2
        w = weights.view(1, -1, 1)
        return (se * w).mean()

    def _extract_student_outputs(self, model_out):
        # Handles outputs with or without attention/features
        if isinstance(model_out, tuple):
            if len(model_out) == 3:
                return model_out[0], model_out[1], model_out[2]
            elif len(model_out) == 2:
                return model_out[0], model_out[1], None
        return model_out, None, None

    def _attn_to_corr(self, attns):
        # attns: list of [B, H, N, N] or None
        if attns is None or len(attns) == 0 or attns[0] is None:
            return None
        # average over layers and heads
        maps = []
        for a in attns:
            if a is None:
                continue
            # a: [B, H, N, N]
            if a.dim() == 4:
                maps.append(a.mean(dim=1))  # [B, N, N]
        if not maps:
            return None
        corr = torch.stack(maps, dim=0).mean(dim=0)  # [B, N, N]
        return corr

    def _features_to_corr(self, feats):
        # feats: [B, N, E]
        if feats is None:
            return None
        f = torch.nn.functional.normalize(feats, p=2, dim=-1)
        return torch.matmul(f, f.transpose(-1, -2))  # [B, N, N]

    def _outputs_to_corr(self, outputs):
        # outputs: [B, L, N] -> corr over horizon dimension
        if outputs is None:
            return None
        # transpose to [B, N, L]
        x = outputs.transpose(1, 2)
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        return torch.matmul(x, x.transpose(-1, -2))  # [B, N, N]

    # 实例化模型
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    # 调用 data_provider 返回数据集和加载器
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    # 创建优化器
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    # 设置损失函数
    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    # 验证循环
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    # 完整训练流程
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

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # prepare teacher if enabled (for stage 2)
        self.teacher = self._build_teacher_model()
        lambda_f = float(getattr(self.args, 'lambda_feature', 0.0))
        lambda_c = float(getattr(self.args, 'lambda_corr', 0.0))
        lambda_o = float(getattr(self.args, 'lambda_out', 0.0))
        sup_epochs = int(getattr(self.args, 'supervised_epochs', 0))
        if getattr(self.args, 'enable_distill', False) and sup_epochs <= 0:
            sup_epochs = max(1, self.args.train_epochs // 2)
        pred_weights = self._get_horizon_weights(self.args.pred_len, self.device)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder-only forward; no injection to decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        out_s = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        s_out, s_attn, s_feat = self._extract_student_outputs(out_s)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        s_out = s_out[:, -self.args.pred_len:, f_dim:]
                        t_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                        # stage schedule
                        if epoch < sup_epochs or not getattr(self.args, 'enable_distill', False):
                            loss = self._weighted_mse(s_out, t_y, pred_weights)
                        else:
                            # teacher forward (frozen)
                            with torch.no_grad():
                                out_t = self.teacher(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                                t_out, t_attn, t_feat = self._extract_student_outputs(out_t)
                                t_out = t_out[:, -self.args.pred_len:, f_dim:]
                            # losses
                            L_sup = self._weighted_mse(s_out, t_y, pred_weights)
                            L_out = self._weighted_mse(s_out, t_out, pred_weights) if lambda_o > 0 else 0.0
                            # feature alignment if dims match
                            if lambda_f > 0 and (s_feat is not None) and (t_feat is not None) and (s_feat.shape[-1] == t_feat.shape[-1]):
                                L_feat = torch.mean((s_feat - t_feat) ** 2)
                            else:
                                L_feat = 0.0
                            # corr alignment via attn -> corr -> mse, fallback to features or outputs
                            S_corr = self._attn_to_corr(s_attn)
                            T_corr = self._attn_to_corr(t_attn)
                            if (S_corr is None) or (T_corr is None):
                                S_corr = self._features_to_corr(s_feat) if s_feat is not None else None
                                T_corr = self._features_to_corr(t_feat) if t_feat is not None else None
                            if (S_corr is None) or (T_corr is None):
                                S_corr = self._outputs_to_corr(s_out)
                                T_corr = self._outputs_to_corr(t_out)
                            if lambda_c > 0 and (S_corr is not None) and (T_corr is not None):
                                L_corr = torch.mean((S_corr - T_corr) ** 2)
                            else:
                                L_corr = 0.0
                            loss = L_sup + lambda_f * L_feat + lambda_c * L_corr + lambda_o * L_out
                        train_loss.append(float(loss.item() if isinstance(loss, torch.Tensor) else loss))
                else:
                    out_s = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    s_out, s_attn, s_feat = self._extract_student_outputs(out_s)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    s_out = s_out[:, -self.args.pred_len:, f_dim:]
                    t_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    if epoch < sup_epochs or not getattr(self.args, 'enable_distill', False):
                        loss = self._weighted_mse(s_out, t_y, pred_weights)
                    else:
                        with torch.no_grad():
                            out_t = self.teacher(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            t_out, t_attn, t_feat = self._extract_student_outputs(out_t)
                            t_out = t_out[:, -self.args.pred_len:, f_dim:]
                        L_sup = self._weighted_mse(s_out, t_y, pred_weights)
                        L_out = self._weighted_mse(s_out, t_out, pred_weights) if lambda_o > 0 else 0.0
                        if lambda_f > 0 and (s_feat is not None) and (t_feat is not None) and (s_feat.shape[-1] == t_feat.shape[-1]):
                            L_feat = torch.mean((s_feat - t_feat) ** 2)
                        else:
                            L_feat = 0.0
                        S_corr = self._attn_to_corr(s_attn)
                        T_corr = self._attn_to_corr(t_attn)
                        if (S_corr is None) or (T_corr is None):
                            S_corr = self._features_to_corr(s_feat) if s_feat is not None else None
                            T_corr = self._features_to_corr(t_feat) if t_feat is not None else None
                        if (S_corr is None) or (T_corr is None):
                            S_corr = self._outputs_to_corr(s_out)
                            T_corr = self._outputs_to_corr(t_out)
                        if lambda_c > 0 and (S_corr is not None) and (T_corr is not None):
                            L_corr = torch.mean((S_corr - T_corr) ** 2)
                        else:
                            L_corr = 0.0
                        loss = L_sup + lambda_f * L_feat + lambda_c * L_corr + lambda_o * L_out
                    train_loss.append(float(loss.item() if isinstance(loss, torch.Tensor) else loss))

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    # 测试
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    # 预测
    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
