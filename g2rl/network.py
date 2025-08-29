import torch
import torch.nn as nn
import torch.nn.functional as F
# ✅ 添加这一行以支持 Python 3.9
from typing import Optional, List


# class CRNNModel(nn.Module):
#     def __init__(
#             self,
#             num_actions: int = 5,
#             num_timesteps: int = 4,
#             initial_channels: int = 4,
#             hidden_size: int = 128,
#             lstm_input_size: Optional[int] = None,
#             num_kernels: Optional[List[int]] = None
#         ):
#         super(CRNNModel, self).__init__()
#         if num_kernels is None:
#             num_kernels = [32, 64]
#         self.num_timesteps = num_timesteps
#         self.num_actions = num_actions
#         self.hidden_size = hidden_size
#         self.initial_channels = initial_channels

#         # CNN blocks
#         self.conv_blocks = nn.ModuleList()
#         in_channels = initial_channels
#         for out_channels in num_kernels:
#             block = nn.Sequential(
#                 nn.Conv3d(
#                     in_channels=in_channels,
#                     out_channels=out_channels,
#                     kernel_size=(1, 3, 3),
#                     stride=(1, 1, 1),
#                     padding=(0, 1, 1)),
#                 nn.ReLU(),
#                 nn.Conv3d(
#                     in_channels=out_channels,
#                     out_channels=out_channels,
#                     kernel_size=(1, 3, 3),
#                     stride=(1, 2, 2),
#                     padding=(0, 1, 1)),
#                 nn.ReLU()
#             )
#             self.conv_blocks.append(block)
#             in_channels = out_channels

#         # LSTM Layer
#         # if lstm_input_size is None:
#         #     self.lstm = None
#         # else:
#         #     self.lstm = nn.LSTM(
#         #         input_size=lstm_input_size,
#         #         hidden_size=hidden_size,
#         #         batch_first=True)
#         self.lstm = None if lstm_input_size is None else nn.LSTM(
#             input_size=lstm_input_size,
#             hidden_size=hidden_size,
#             batch_first=True
#         )

#         # FC layers
#         self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
#         self.fc2 = nn.Linear(hidden_size // 2, num_actions)

#     def forward(self, x):
#         # reshape to (batch_size, channels, depth, height, width)
#         x = x.permute(0, 4, 1, 2, 3)

#         for block in self.conv_blocks:
#             x = block(x)

#         # determine the LSTM input size if it hasn't been set
#         if self.lstm is None:
#             batch_size, _, depth, height, width = x.size()
#             lstm_input_size = height * width * self.conv_blocks[-1][0].out_channels
#             self.lstm = nn.LSTM(
#                 input_size=lstm_input_size,
#                 hidden_size=self.hidden_size,
#                 batch_first=True,
#                 device=x.device)

#         batch_size, _, depth, height, width = x.size()
#         x = x.reshape(batch_size, self.num_timesteps, -1)

#         lstm_out, _ = self.lstm(x)
#         x = lstm_out[:, -1, :]

#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# g2rl/network.py
import torch
import torch.nn as nn

class CRNNModel(nn.Module):
    def __init__(self, num_actions: int = 5, in_channels: int = 11):
        super().__init__()
        self.adapt = nn.Conv3d(in_channels, 32, kernel_size=1)  # 固定输入通道
        self.backbone = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=(3,3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,2,2)),
        )
        self.pool = nn.AdaptiveAvgPool3d((4, 2, 5))
        self.fc1  = nn.LazyLinear(128)
        self.fc2  = nn.Linear(128, num_actions)

    def forward(self, x):
        x = self.adapt(x)   # 期望 x 形状 [N, in_channels, D, H, W]
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

