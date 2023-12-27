import math
import torch
import torch.nn as nn
from typing import Optional

from nd.models.brain_encoder import TemporalAggregation


class EEGNetDeep(nn.Module):
    def __init__(self, args, duration):
        super(EEGNetDeep, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, args.F1, (1, args.k1), padding="same", bias=False),
            nn.BatchNorm2d(args.F1),
        )

        if args.use_dilation:
            self.conv2 = nn.Sequential(
                nn.Conv2d(
                    args.F1,
                    args.D * args.F1,
                    (args.num_channels // args.num_channels_per_patch, 1),
                    groups=args.F1,
                    bias=False,
                    dilation=(args.num_channels_per_patch, 1),
                ),
                nn.BatchNorm2d(args.D * args.F1),
                nn.GELU(),
                # nn.AvgPool2d((1, args.p1)),  # 2
                # nn.Dropout(args.dr1),
            )
        else:
            self.conv2 = nn.Sequential(
                nn.Conv2d(
                    args.F1,
                    args.D * args.F1,
                    (args.num_channels_per_patch, 1),
                    groups=args.F1,
                    bias=False,
                    stride=(args.num_channels_per_patch, 1),
                ),
                nn.BatchNorm2d(args.D * args.F1),
                nn.GELU(),
                # nn.AvgPool2d((1, args.p1)),  # 2
                # nn.Dropout(args.dr1),
            )

        # 電極方向の時限1にする
        residual_spatial_dim = self.residual_spatial_dim(args.num_channels, duration)
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                args.D * args.F1,
                args.D * args.F1,
                (residual_spatial_dim, 1),
                groups=args.F1,
                bias=False,
                stride=(1, args.stride1),  # (1, 2)
            ),
            nn.BatchNorm2d(args.D * args.F1),
            nn.GELU(),
            # nn.AvgPool2d((1, args.p1)),  # 2
            nn.Dropout(args.dr1),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                args.D * args.F1,
                args.D * args.F1,
                (1, args.k2),
                padding=0,
                groups=args.D * args.F1,
                bias=False,
                stride=(1, args.stride2),  # (1, 2)
            ),
            nn.Conv2d(args.D * args.F1, args.F2, (1, 1), bias=False),
            nn.BatchNorm2d(args.F2),
            nn.GELU(),
            # nn.AvgPool2d((1, args.p2)),  # 4
            nn.Dropout(args.dr2),
        )

        ret_k = self.compute_time_kernel_len(
            args.num_channels, duration, args.t_mel, args.k2
        )
        if type(ret_k) is tuple:
            k_len, stride = ret_k
            self.conv_trans = nn.ConvTranspose1d(
                args.F2,
                args.F2,
                args.k2,
                stride=stride,
                padding=0,
                output_padding=0,
                dilation=1,
                bias=False,
            )
        else:
            k_len = ret_k
            self.conv_trans = None
        self.conv_time1 = nn.Conv1d(
            args.F2, args.F2, k_len, padding=0, dilation=1, stride=1, bias=True
        )
        time_layers_list = []
        for _ in range(args.num_conv_time_layers - 1):
            time_layers_list += [
                nn.GELU(),
                nn.Conv1d(
                    args.F2,
                    args.F2,
                    self.nearest_odd_quotient(k_len, args.k_div),
                    padding="same",
                    dilation=1,
                    stride=1,
                    bias=False,
                ),
            ]
        self.conv_time2 = nn.Sequential(*time_layers_list)
        self.conv_emb1 = nn.Conv1d(args.F2, args.n_mel, 1, bias=True)
        emb_layers_list = []
        for _ in range(args.num_conv_emb_layers - 1):
            emb_layers_list += [
                nn.GELU(),
                nn.Conv1d(args.n_mel, args.n_mel, 1, bias=False),
            ]
        self.conv_emb2 = nn.Sequential(*emb_layers_list)

        self.temporal_aggregation = TemporalAggregation(args, args.t_mel)

    def forward(self, x):
        x = x.unsqueeze(
            dim=1
        )  # if x shape is (B, C, T), then x shape is (B, 1, C, T) (B, 1, 128, 1440)
        x = self.conv1(x)  # 1, 16, 128, 1440
        x = self.conv2(x)  # 1, 16, 8, 720
        x = self.conv3(x)  # 1, 16, 1, 720
        x = self.conv4(x)  # 1, 32, 1, 360
        x = x.squeeze(dim=2)  # (B, C, T)
        if self.conv_trans:
            x = self.conv_trans(x)
        x = self.conv_time1(x)
        x = self.conv_time2(x)
        x = self.conv_emb1(x)
        x = self.conv_emb2(x)

        return self.temporal_aggregation(x)

    def residual_spatial_dim(self, num_channels, duration):
        x = torch.zeros((1, 1, num_channels, duration))
        x = self.conv1(x)
        x = self.conv2(x)
        size = x.size()  # (B, C, Ch, T)
        return size[2]

    def compute_time_kernel_len(self, num_channels, duration, t_mel, k_trans):
        x = torch.zeros((1, 1, num_channels, duration))
        x = self.conv1(x)  # 1, 16, 128, 1440
        x = self.conv2(x)  # 1, 16, 8, 720
        x = self.conv3(x)  # 1, 16, 1, 720
        x = self.conv4(x)  # 1, 32, 1, 360
        x = x.squeeze(dim=2)  # (B, C, T)
        _, C, T = x.size()
        if T - t_mel + 1 > 0:
            return T - t_mel + 1
        else:
            stride = math.ceil((t_mel - 1 - k_trans) / (T - 1))
            T = (T - 1) * stride + k_trans
            return (T - t_mel + 1, stride)

    @staticmethod
    def nearest_odd_quotient(n, m):
        quotient = n // m
        if quotient % 2 == 0:  # 商が偶数の場合
            if abs(n - (quotient * m)) >= abs(n - ((quotient + 1) * m)):
                return (quotient + 1) * m
            else:
                return quotient * m
        else:  # 商が奇数の場合
            if abs(n - (quotient * m)) >= abs(n - ((quotient - 1) * m)):
                return (quotient - 1) * m
            else:
                return quotient * m
