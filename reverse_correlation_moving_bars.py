#!/usr/bin/env python3
"""
反向关联实验：使用黑白色块移动动画替代随机白噪声作为刺激，计算神经元感受野。
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
from fnn import microns
from matplotlib.font_manager import FontProperties

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'PingFang SC']  # 优先使用这些字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 创建输出目录
output_dir = "receptive_fields_moving_bars"
os.makedirs(output_dir, exist_ok=True)

def generate_moving_bar_stimulus(n_frames=30, height=144, width=256, direction='horizontal', speed=4, 
                                contrast=50, background=128, bar_color=0):
    """
    生成移动的黑白条形刺激
    
    Args:
        n_frames: 帧数
        height: 图像高度
        width: 图像宽度
        direction: 'horizontal'（水平移动）、'vertical'（垂直移动）、'diagonal_down'（对角线向下）或'diagonal_up'（对角线向上）
        speed: 每帧移动的像素数
        contrast: 对比度控制（影响背景与条形的差异）
        background: 背景灰度值（0-255）
        bar_color: 条形灰度值（0-255）
    
    Returns:
        形状为[n_frames, height, width]的numpy数组，元素为uint8类型（0-255）
    """
    # 确保颜色值在有效范围内
    background = np.clip(background, 0, 255)
    bar_color = np.clip(bar_color, 0, 255)
    
    # 初始化刺激数组为背景色
    stimuli = np.ones((n_frames, height, width), dtype=np.uint8) * background
    
    # 条形宽度（屏幕宽度/高度的10%）
    bar_width = int(width * 0.1) if direction in ['horizontal', 'diagonal_down', 'diagonal_up'] else int(height * 0.1)
    
    # 考虑重复移动，使条形可以多次穿过屏幕
    max_distance = max(width, height) * 2  # 足够长的距离以保证多次穿过
    starting_positions = np.arange(-bar_width, max_distance, max_distance//3)[:3]  # 取3个不同的起始位置
    
    for start_pos in starting_positions:
        for i in range(n_frames):
            if direction == 'horizontal':
                # 水平移动的条形，从左向右
                pos = (start_pos + i * speed) % (width + bar_width * 2) - bar_width
                if -bar_width < pos < width:
                    # 计算条形在画面内的部分
                    start_x = max(0, pos)
                    end_x = min(width, pos + bar_width)
                    stimuli[i, :, start_x:end_x] = bar_color
            
            elif direction == 'vertical':
                # 垂直移动的条形，从上向下
                pos = (start_pos + i * speed) % (height + bar_width * 2) - bar_width
                if -bar_width < pos < height:
                    # 计算条形在画面内的部分
                    start_y = max(0, pos)
                    end_y = min(height, pos + bar_width)
                    stimuli[i, start_y:end_y, :] = bar_color
            
            elif direction == 'diagonal_down':
                # 对角线移动（左上到右下）
                pos_x = (start_pos + i * speed) % (width + bar_width * 2) - bar_width
                pos_y = (start_pos + i * speed) % (height + bar_width * 2) - bar_width
                for j in range(bar_width):
                    x = pos_x + j
                    y = pos_y + j
                    if 0 <= x < width and 0 <= y < height:
                        stimuli[i, y, x] = bar_color
                        # 使对角线更宽一些，便于识别
                        for k in range(1, 3):
                            if y+k < height: stimuli[i, y+k, x] = bar_color
                            if y-k >= 0: stimuli[i, y-k, x] = bar_color
                            if x+k < width: stimuli[i, y, x+k] = bar_color
                            if x-k >= 0: stimuli[i, y, x-k] = bar_color
            
            elif direction == 'diagonal_up':
                # 对角线移动（左下到右上）
                pos_x = (start_pos + i * speed) % (width + bar_width * 2) - bar_width
                pos_y = height - ((start_pos + i * speed) % (height + bar_width * 2) - bar_width)
                for j in range(bar_width):
                    x = pos_x + j
                    y = pos_y - j
                    if 0 <= x < width and 0 <= y < height:
                        stimuli[i, y, x] = bar_color
                        # 使对角线更宽一些，便于识别
                        for k in range(1, 3):
                            if y+k < height: stimuli[i, y+k, x] = bar_color
                            if y-k >= 0: stimuli[i, y-k, x] = bar_color
                            if x+k < width: stimuli[i, y, x+k] = bar_color
                            if x-k >= 0: stimuli[i, y, x-k] = bar_color
                
    return stimuli

def generate_diverse_stimuli(n_frames=1000, height=144, width=256, contrast=50):
    """
    生成多样化的黑白条形刺激序列，包含不同方向的运动条形
    
    Args:
        n_frames: 总帧数
        height: 图像高度
        width: 图像宽度
        contrast: 对比度
        
    Returns:
        形状为[n_frames, height, width]的numpy数组
    """
    # 确定每个方向分配的帧数
    frames_per_direction = n_frames // 4
    remaining_frames = n_frames - (frames_per_direction * 4)
    
    # 生成四种不同方向的刺激
    h_stimuli = generate_moving_bar_stimulus(n_frames=frames_per_direction, height=height, width=width, 
                                           direction='horizontal', speed=4, contrast=contrast)
    v_stimuli = generate_moving_bar_stimulus(n_frames=frames_per_direction, height=height, width=width, 
                                           direction='vertical', speed=4, contrast=contrast)
    d_down_stimuli = generate_moving_bar_stimulus(n_frames=frames_per_direction, height=height, width=width, 
                                               direction='diagonal_down', speed=4, contrast=contrast)
    d_up_stimuli = generate_moving_bar_stimulus(n_frames=frames_per_direction, height=height, width=width, 
                                             direction='diagonal_up', speed=4, contrast=contrast)
    
    # 如果有剩余帧，添加到水平方向
    if remaining_frames > 0:
        extra_stimuli = generate_moving_bar_stimulus(n_frames=remaining_frames, height=height, width=width, 
                                                  direction='horizontal', speed=4, contrast=contrast)
        # 组合所有刺激
        all_stimuli = np.concatenate([h_stimuli, v_stimuli, d_down_stimuli, d_up_stimuli, extra_stimuli])
    else:
        # 组合所有刺激
        all_stimuli = np.concatenate([h_stimuli, v_stimuli, d_down_stimuli, d_up_stimuli])
    
    # 随机打乱帧的顺序，使刺激更多样化
    np.random.shuffle(all_stimuli)
    
    return all_stimuli

def compute_reverse_correlation_batch(model, neuron_indices, n_frames=1000, batch_size=50, contrast=50, n_lags=1):
    """
    使用黑白条形刺激计算一批神经元的反向关联（spike-triggered average）。
    
    Args:
        model: FNN模型
        neuron_indices: 要分析的神经元索引
        n_frames: 总帧数
        batch_size: 每批次处理的帧数
        contrast: 对比度
        n_lags: 时间滞后的帧数
        
    Returns:
        dict: 神经元索引到其STA结果的映射
    """
    results = {}
    n_batches = n_frames // batch_size
    
    # 初始化用于存储每个时间滞后的加权和和总响应的数组
    weighted_sums = {idx: np.zeros((n_lags, 144, 256)) for idx in neuron_indices}
    total_responses = {idx: 0.0 for idx in neuron_indices}
    
    for i in range(n_batches):
        # 生成多样化的黑白条形刺激（而不是白噪声）
        stimuli = generate_diverse_stimuli(n_frames=batch_size + n_lags - 1, contrast=contrast)
        
        # 零均值版本，用于STA计算
        stimuli_zero_mean = stimuli.astype(np.float32) - 128.0
        
        # 获取所有神经元的模型响应
        responses = model.predict(stimuli=stimuli)
        
        # 使用向量化操作处理每个神经元的响应
        # 对每个神经元累积时空STA
        for idx in neuron_indices:
            neuron_response = responses[n_lags-1:, idx]
            weighted_sum = np.zeros((n_lags, 144, 256))
            for j, resp in enumerate(neuron_response):
                for lag in range(n_lags):
                    weighted_sum[lag] += resp * stimuli_zero_mean[j + n_lags - 1 - lag]
            
            weighted_sums[idx] += weighted_sum
            total_responses[idx] += neuron_response.sum()
    
    # 通过总响应归一化获取STA
    for idx in neuron_indices:
        if total_responses[idx] > 0:
            results[idx] = weighted_sums[idx] / total_responses[idx]
        else:
            results[idx] = weighted_sums[idx]
    
    return results

def compute_reverse_correlation(model, neuron_indices, n_frames=1000, batch_size=50, contrast=50, n_lags=1, 
                               n_jobs=1, silent=False):
    """
    计算神经元的反向关联（spike-triggered average）。
    
    Args:
        model: FNN模型
        neuron_indices: 要分析的神经元索引列表
        n_frames: 用于分析的总帧数
        batch_size: 每批处理的帧数
        contrast: 对比度
        n_lags: 时间滞后的帧数
        n_jobs: 并行作业数
        silent: 是否禁用打印信息
        
    Returns:
        dict: 神经元索引到其STA结果的映射
    """
    if n_jobs <= 1:
        result = compute_reverse_correlation_batch(model, neuron_indices, n_frames, batch_size, contrast, n_lags)
    else:
        # 并行计算STA
        if not silent:
            print(f"使用 {n_jobs} 个工作进程并行计算反向关联...")
        
        # 将神经元分成多个批次
        batch_size_neurons = max(1, len(neuron_indices) // n_jobs)
        neuron_batches = [neuron_indices[i:i+batch_size_neurons] 
                         for i in range(0, len(neuron_indices), batch_size_neurons)]
        
        # 使用进程池并行计算
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            func = partial(compute_reverse_correlation_batch, model, 
                          n_frames=n_frames, batch_size=batch_size, 
                          contrast=contrast, n_lags=n_lags)
            results = list(executor.map(func, neuron_batches))
        
        # 合并结果
        result = {}
        for res in results:
            result.update(res)
    
    if not silent:
        print(f"已完成 {len(neuron_indices)} 个神经元的计算")
    
    return result

def visualize_receptive_field(sta, neuron_idx, file_path=None):
    """
    可视化感受野并保存到文件。
    
    Args:
        sta: 尖峰触发平均图像
        neuron_idx: 被分析的神经元的索引
        file_path: 保存可视化的路径
    """
    # STA可能包含多个时间滞后
    sta = np.asarray(sta)
    if sta.ndim == 2:
        sta = sta[np.newaxis, ...]
    
    n_lags = sta.shape[0]
    plt.figure(figsize=(5 * n_lags, 4))
    
    for i in range(n_lags):
        plt.subplot(1, n_lags, i + 1)
        sta_norm = (sta[i] - np.mean(sta[i])) / np.std(sta[i])
        plt.imshow(sta_norm, cmap='coolwarm')
        plt.title(f'滞后 {i}')
        plt.axis('off')
    
    plt.suptitle(f'神经元 {neuron_idx} 的感受野特征图像')
    
    if file_path:
        plt.savefig(file_path)
        plt.close()  # 关闭图表以释放内存
    else:
        plt.show()

def analyze_neuron_properties(model, ids, neuron_indices):
    """
    分析并返回所选神经元的属性。
    
    Args:
        model: FNN模型
        ids: 神经元ID数据框
        neuron_indices: 要分析的神经元索引
        
    Returns:
        dict: 包含神经元属性的字典
    """
    properties = {}
    
    # 为简单起见，我们只返回神经元索引和类型（如果有的话）
    for idx in neuron_indices:
        neuron_id = ids.iloc[idx].unit_id if idx < len(ids) else None
        properties[idx] = {
            'unit_id': neuron_id,
            'type': ids.iloc[idx].cell_type if 'cell_type' in ids.columns and idx < len(ids) else None
        }
    
    return properties

def save_receptive_fields(result, output_dir, neuron_indices=None):
    """
    保存感受野图像到指定目录。
    
    Args:
        result: 包含STA结果的字典
        output_dir: 输出目录
        neuron_indices: 要保存的神经元索引列表，如果为None则保存所有结果
    """
    if neuron_indices is None:
        neuron_indices = list(result.keys())
    
    os.makedirs(output_dir, exist_ok=True)
    
    for idx in neuron_indices:
        sta = result[idx]
        file_path = os.path.join(output_dir, f"receptive_field_{idx}.png")
        visualize_receptive_field(sta, idx, file_path)

def load_model_and_select_neurons(n_neurons=10, min_activation=0.1):
    """
    加载模型并选择最活跃的神经元。
    
    Args:
        n_neurons: 要选择的神经元数量
        min_activation: 最小激活阈值
        
    Returns:
        tuple: (model, ids, selected_neuron_indices)
    """
    print("加载模型...")
    start_time = time.time()
    model, ids = microns.scan(session=8, scan_idx=5, verbose=True)
    load_time = time.time() - start_time
    print(f"模型在 {load_time:.2f} 秒内加载完成")
    
    print("使用CPU进行计算，因为MPS与grid_sample边界填充有兼容性问题")
    
    # 打印基本模型信息
    n_total_neurons = len(ids)
    print(f"模型中的总神经元数：{n_total_neurons}")
    
    print("运行测试以识别最活跃的神经元...")
    # 生成测试刺激并测量神经元激活
    n_test_frames = 100
    test_stimuli = generate_diverse_stimuli(n_frames=n_test_frames, contrast=80)
    
    # 获取所有神经元对测试刺激的模型响应
    test_responses = model.predict(stimuli=test_stimuli)
    
    # 计算每个神经元的激活强度（平均响应）
    activation_strength = np.mean(test_responses, axis=0)
    
    # 按激活强度对神经元进行排序
    sorted_indices = np.argsort(-activation_strength)  # 降序排序
    
    # 选择超过阈值的神经元中最活跃的n_neurons个
    selected_indices = [idx for idx in sorted_indices if activation_strength[idx] > min_activation][:n_neurons]
    
    print(f"已选择 {len(selected_indices)} 个活跃神经元，索引：{selected_indices[:10]}...")
    
    return model, ids, selected_indices

def visualize_stimulus_examples(save_path=None):
    """
    可视化不同方向的移动条形刺激示例。
    
    Args:
        save_path: 保存图像的路径
    """
    directions = ['horizontal', 'vertical', 'diagonal_down', 'diagonal_up']
    plt.figure(figsize=(16, 4 * len(directions)))
    
    for i, direction in enumerate(directions):
        stimuli = generate_moving_bar_stimulus(n_frames=30, direction=direction)
        indices = np.linspace(0, 29, 6, dtype=int)  # 从30帧中均匀选择6帧
        
        for j, idx in enumerate(indices):
            plt.subplot(len(directions), 6, i * 6 + j + 1)
            plt.imshow(stimuli[idx], cmap='gray')
            if j == 0:
                plt.ylabel(direction)
            plt.title(f'帧 {idx}')
            plt.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def compare_receptive_fields(rf_white_noise_dir, rf_moving_bars_dir, selected_neurons, n_neurons_to_show=5, save_path=None):
    """
    比较使用白噪声和移动条形刺激得到的感受野。
    
    Args:
        rf_white_noise_dir: 白噪声感受野目录
        rf_moving_bars_dir: 移动条形感受野目录
        selected_neurons: 要比较的神经元列表
        n_neurons_to_show: 要显示的神经元数量
        save_path: 保存图像的路径
    """
    neurons_to_show = selected_neurons[:n_neurons_to_show]
    
    plt.figure(figsize=(12, 3 * n_neurons_to_show))
    
    for i, neuron_idx in enumerate(neurons_to_show):
        # 加载白噪声感受野
        wn_path = os.path.join(rf_white_noise_dir, f"receptive_field_{neuron_idx}.png")
        wn_rf = plt.imread(wn_path) if os.path.exists(wn_path) else None
        
        # 加载移动条形感受野
        mb_path = os.path.join(rf_moving_bars_dir, f"receptive_field_{neuron_idx}.png")
        mb_rf = plt.imread(mb_path) if os.path.exists(mb_path) else None
        
        # 显示两种方法的结果
        if wn_rf is not None:
            plt.subplot(n_neurons_to_show, 2, i * 2 + 1)
            plt.imshow(wn_rf)
            plt.title(f'神经元 {neuron_idx} - 白噪声')
            plt.axis('off')
        
        if mb_rf is not None:
            plt.subplot(n_neurons_to_show, 2, i * 2 + 2)
            plt.imshow(mb_rf)
            plt.title(f'神经元 {neuron_idx} - 移动条形')
            plt.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def main():
    """主函数，运行移动条形刺激的反向关联实验。"""
    print("开始使用移动条形刺激的FNN反向关联分析...")
    total_start_time = time.time()
    
    # 设置随机种子以获得可重复的结果
    np.random.seed(42)
    
    # 可视化刺激示例
    print("生成并可视化刺激示例...")
    stimulus_examples_path = os.path.join(output_dir, "stimulus_examples.png")
    visualize_stimulus_examples(save_path=stimulus_examples_path)
    print(f"刺激示例已保存至 {stimulus_examples_path}")
    
    # 加载模型并选择神经元
    model, ids, selected_neuron_indices = load_model_and_select_neurons(n_neurons=20)
    
    # 确定要分析的帧数和批大小
    n_frames = 2000  # 总帧数
    batch_size = 50  # 每批次处理的帧数
    n_lags = 1      # 时间滞后的帧数
    contrast = 50    # 对比度
    
    print(f"使用移动条形刺激计算神经元的反向关联...")
    print(f"参数: {n_frames} 帧, 批大小 {batch_size}, 对比度 {contrast}")
    
    # 由于模型无法序列化（pickle），使用单线程计算
    n_jobs = 1
    
    # 计算反向关联
    sta_start_time = time.time()
    result = compute_reverse_correlation(
        model, selected_neuron_indices, 
        n_frames=n_frames, batch_size=batch_size, 
        contrast=contrast, n_lags=n_lags,
        n_jobs=n_jobs
    )
    sta_time = time.time() - sta_start_time
    print(f"反向关联计算完成，用时 {sta_time:.2f} 秒")
    
    # 可视化并保存结果
    print("可视化并保存感受野图像...")
    save_receptive_fields(result, output_dir, selected_neuron_indices)
    
    # 如果之前的白噪声实验结果存在，比较两种方法
    white_noise_dir = "receptive_fields"
    if os.path.exists(white_noise_dir):
        print("比较白噪声和移动条形刺激的结果...")
        comparison_path = os.path.join(output_dir, "comparison.png")
        compare_receptive_fields(white_noise_dir, output_dir, selected_neuron_indices, 
                                n_neurons_to_show=5, save_path=comparison_path)
        print(f"比较图已保存至 {comparison_path}")
    
    total_time = time.time() - total_start_time
    print(f"全部反向关联实验完成，总用时 {total_time:.2f} 秒")

if __name__ == "__main__":
    main()
