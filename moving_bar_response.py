#!/usr/bin/env python3
"""
使用黑白色块移动的动画刺激FNN模型，并以瀑布图形式可视化神经元响应。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from fnn import microns
import time
from matplotlib.font_manager import FontProperties

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'PingFang SC']  # 优先使用这些字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

def generate_moving_bar_stimulus(n_frames=30, height=144, width=256, direction='horizontal', speed=8):
    """
    生成移动的黑白条形刺激
    
    Args:
        n_frames: 帧数 (对应1秒的刺激，假设fps=30)
        height: 图像高度
        width: 图像宽度
        direction: 'horizontal'（水平移动）或'vertical'（垂直移动）
        speed: 每帧移动的像素数
    
    Returns:
        形状为[n_frames, height, width]的numpy数组，元素为uint8类型（0-255）
    """
    stimuli = np.ones((n_frames, height, width), dtype=np.uint8) * 128  # 中灰背景
    
    # 条形宽度（20%的宽度或高度）
    bar_width = int(width * 0.2) if direction == 'horizontal' else int(height * 0.2)
    
    for i in range(n_frames):
        if direction == 'horizontal':
            # 水平移动的黑条，初始位置在左侧之外，向右移动
            pos = -bar_width + i * speed
            if pos < width:
                # 计算条形在画面内的部分
                start_x = max(0, pos)
                end_x = min(width, pos + bar_width)
                stimuli[i, :, start_x:end_x] = 0  # 黑色条形
        else:
            # 垂直移动的黑条，初始位置在顶部之外，向下移动
            pos = -bar_width + i * speed
            if pos < height:
                # 计算条形在画面内的部分
                start_y = max(0, pos)
                end_y = min(height, pos + bar_width)
                stimuli[i, start_y:end_y, :] = 0  # 黑色条形
                
    return stimuli

def visualize_neuron_responses(responses, selected_neurons=None, n_neurons=50, title='神经元响应瀑布图'):
    """
    创建神经元响应的瀑布图
    
    Args:
        responses: 神经元响应数组，形状为[n_frames, n_neurons]
        selected_neurons: 要可视化的特定神经元的索引列表，如果为None则选择响应最强的n_neurons个
        n_neurons: 如果selected_neurons为None，则显示响应最强的n_neurons个
        title: 图表标题
    """
    if selected_neurons is None:
        # 计算每个神经元的响应强度（最大响应 - 最小响应）
        response_strength = np.max(responses, axis=0) - np.min(responses, axis=0)
        # 选择响应最强的n_neurons个神经元
        selected_neurons = np.argsort(-response_strength)[:n_neurons]
    else:
        n_neurons = len(selected_neurons)
    
    plt.figure(figsize=(10, 8))
    
    # 创建瀑布图
    for i, neuron_idx in enumerate(selected_neurons):
        # 获取该神经元的响应
        neuron_response = responses[:, neuron_idx]
        # 归一化到[0, 1]区间，便于可视化
        neuron_response_norm = (neuron_response - np.min(neuron_response)) / (np.max(neuron_response) - np.min(neuron_response) + 1e-10)
        # 画线，并向上偏移以形成瀑布效果
        plt.plot(neuron_response_norm + i, linewidth=1)
    
    plt.title(title)
    plt.ylabel('神经元索引')
    plt.xlabel('时间 (帧)')
    plt.yticks(np.arange(0, n_neurons, 5), [selected_neurons[i] for i in range(0, n_neurons, 5)])
    plt.tight_layout()
    
    # 保存图片
    output_dir = 'experiment_results'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'{output_dir}/neuron_waterfall_{timestamp}.png')
    print(f"瀑布图已保存至 {output_dir}/neuron_waterfall_{timestamp}.png")
    plt.show()

def visualize_stimulus(stimulus, save=True):
    """
    可视化生成的刺激序列
    
    Args:
        stimulus: 刺激序列，形状为[n_frames, height, width]
        save: 是否保存图像
    """
    n_frames = stimulus.shape[0]
    # 选择均匀分布的6帧显示
    indices = np.linspace(0, n_frames-1, 6, dtype=int)
    
    plt.figure(figsize=(12, 4))
    for i, idx in enumerate(indices):
        plt.subplot(1, 6, i+1)
        plt.imshow(stimulus[idx], cmap='gray')
        plt.title(f'帧 {idx}')
        plt.axis('off')
    
    plt.suptitle('黑白色块移动刺激')
    plt.tight_layout()
    
    if save:
        output_dir = 'experiment_results'
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'{output_dir}/stimulus_frames_{timestamp}.png')
        print(f"刺激序列图已保存至 {output_dir}/stimulus_frames_{timestamp}.png")
    
    plt.show()

def main():
    print("正在加载FNN模型...")
    # 使用microns.scan函数加载模型和神经元ID
    model, ids = microns.scan(session=8, scan_idx=5, verbose=True)
    
    # 生成水平移动条形刺激 (30帧=1秒)
    print("生成水平移动条形刺激...")
    horizontal_stimulus = generate_moving_bar_stimulus(n_frames=30, direction='horizontal', speed=8)
    visualize_stimulus(horizontal_stimulus)
    
    # 获取水平移动刺激的神经元响应
    print("计算水平移动刺激的神经元响应...")
    horizontal_responses = model.predict(stimuli=horizontal_stimulus)
    print(f"响应形状: {horizontal_responses.shape}")
    
    # 可视化水平移动刺激的神经元响应
    print("可视化神经元响应...")
    visualize_neuron_responses(horizontal_responses, title='神经元对水平移动条形的响应')
    
    # 生成垂直移动条形刺激
    print("\n生成垂直移动条形刺激...")
    vertical_stimulus = generate_moving_bar_stimulus(n_frames=30, direction='vertical', speed=8)
    visualize_stimulus(vertical_stimulus)
    
    # 获取垂直移动刺激的神经元响应
    print("计算垂直移动刺激的神经元响应...")
    vertical_responses = model.predict(stimuli=vertical_stimulus)
    
    # 可视化垂直移动刺激的神经元响应
    print("可视化神经元响应...")
    visualize_neuron_responses(vertical_responses, title='神经元对垂直移动条形的响应')
    
    # 对比水平和垂直刺激的神经元响应差异
    print("\n分析对水平和垂直移动最敏感的神经元...")
    
    # 计算对水平和垂直移动的响应差异
    horizontal_strength = np.max(horizontal_responses, axis=0) - np.min(horizontal_responses, axis=0)
    vertical_strength = np.max(vertical_responses, axis=0) - np.min(vertical_responses, axis=0)
    
    # 找出对水平移动最敏感的神经元
    horizontal_sensitive = np.where(horizontal_strength > vertical_strength)[0]
    horizontal_sensitive = horizontal_sensitive[np.argsort(-horizontal_strength[horizontal_sensitive])][:20]
    
    # 找出对垂直移动最敏感的神经元
    vertical_sensitive = np.where(vertical_strength > horizontal_strength)[0]
    vertical_sensitive = vertical_sensitive[np.argsort(-vertical_strength[vertical_sensitive])][:20]
    
    print(f"对水平移动最敏感的神经元: {horizontal_sensitive[:10]}...")
    print(f"对垂直移动最敏感的神经元: {vertical_sensitive[:10]}...")
    
    # 可视化对水平移动最敏感的神经元响应
    visualize_neuron_responses(horizontal_responses, selected_neurons=horizontal_sensitive, 
                               title='对水平移动最敏感的神经元响应')
    
    # 可视化对垂直移动最敏感的神经元响应
    visualize_neuron_responses(vertical_responses, selected_neurons=vertical_sensitive, 
                               title='对垂直移动最敏感的神经元响应')

if __name__ == "__main__":
    main()
