import torch
import matplotlib.pyplot as plt
import numpy as np

# 模拟输入参数
B, N, C = 1, 16, 8  # Batch=1, Tokens=16 (4x4), Channels=8
head_dim = 4
patch_resolution = (4, 4)
shift_pixel = 1

# 构造一个可视化的输入：batch=1，16个token，每个token有8个通道
# 我们构造通道部分是 0~7 的固定数字，方便看 shift 变化
input_tensor = torch.zeros((B, N, C))
for i in range(C):
    input_tensor[0, :, i] = i

# 模拟 q_shift_multihead 函数中间的 reshape 部分
input_reshaped = input_tensor.transpose(1, 2).reshape(B, -1, head_dim, patch_resolution[0], patch_resolution[1])
# shape: [1, 2(heads), 4(head_dim), 4(H), 4(W)]

# 做 shift 操作（我们只演示一个 head）
input_head = input_reshaped[0, 0].numpy()  # shape: [4, 4, 4]

# 初始化输出
output_head = np.zeros_like(input_head)

# shift 方向：
output_head[0:1, :, shift_pixel:] = input_head[0:1, :, :-shift_pixel]       # right shift
output_head[1:2, :, :-shift_pixel] = input_head[1:2, :, shift_pixel:]       # left shift
output_head[2:3, shift_pixel:, :] = input_head[2:3, :-shift_pixel, :]       # down shift
output_head[3:4, :-shift_pixel, :] = input_head[3:4, shift_pixel:, :]       # up shift

# 可视化 4 个 channel 的变化前后
fig, axes = plt.subplots(2, 4, figsize=(16, 4))

for i in range(4):
    axes[0, i].imshow(input_head[i], cmap='viridis')
    axes[0, i].set_title(f"Original Channel {i}")
    axes[0, i].axis('off')

    axes[1, i].imshow(output_head[i], cmap='viridis')
    axes[1, i].set_title(f"Shifted Channel {i}")
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()
