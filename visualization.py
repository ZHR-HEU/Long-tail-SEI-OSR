#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scientific Journal-Style Visualization Module for ML Experiments

采用Nature/Science顶级期刊风格:
- 专业配色方案
- 精心设计的标记和线型
- 单栏正方形布局
- 高质量排版
"""

import os
import warnings
from typing import Optional, Dict, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings('ignore', category=UserWarning)


# =============================================================================
# 顶级期刊配色方案 (Nature/Science Style)
# =============================================================================

class JournalColors:
    """
    Nature/Science期刊风格配色
    基于色彩理论和可访问性优化
    """
    # 主色系 - 深邃专业
    DEEP_BLUE = '#2E5090'  # 主要数据线
    VIBRANT_ORANGE = '#E8743B'  # 对比/验证线
    FOREST_GREEN = '#19A979'  # 第三数据系列
    ROYAL_PURPLE = '#A95AA1'  # 第四数据系列
    CRIMSON = '#CC3A3A'  # 强调/警告

    # 辅助色系
    STEEL_GRAY = '#5B6770'  # 辅助信息
    SLATE = '#778899'  # 次要元素
    CHARCOAL = '#2F3640'  # 文字/边框

    # 浅色系 - 用于填充
    LIGHT_BLUE = '#A8C5DD'
    LIGHT_ORANGE = '#F5C6AA'
    LIGHT_GREEN = '#A3DCC4'
    LIGHT_PURPLE = '#D4B5D2'

    # 中性色
    BACKGROUND = '#FFFFFF'
    GRID = '#E5E5E5'

    @classmethod
    def get_main_palette(cls, n: int = 8) -> List[str]:
        """获取主配色板"""
        palette = [
            cls.DEEP_BLUE,
            cls.VIBRANT_ORANGE,
            cls.FOREST_GREEN,
            cls.ROYAL_PURPLE,
            cls.CRIMSON,
            '#6B8E23',  # Olive
            '#4682B4',  # Steel Blue
            '#CD853F',  # Peru
        ]
        if n <= len(palette):
            return palette[:n]
        # 循环使用
        return [palette[i % len(palette)] for i in range(n)]

    @classmethod
    def get_sequential_blues(cls, n: int = 5) -> List[str]:
        """蓝色渐变序列"""
        return ['#C6DBEF', '#9ECAE1', '#6BAED6', '#3182BD', '#08519C'][:n]

    @classmethod
    def get_diverging(cls) -> Tuple[str, str, str]:
        """发散配色：负-中性-正"""
        return cls.VIBRANT_ORANGE, '#F7F7F7', cls.DEEP_BLUE


# =============================================================================
# 标记和线型设计
# =============================================================================

class JournalStyles:
    """期刊级别的视觉样式"""

    # 线型样式（清晰可辨）
    LINE_STYLES = [
        '-',  # 实线
        '--',  # 虚线
        '-.',  # 点划线
        ':',  # 点线
    ]

    # 标记样式（视觉清晰）
    MARKERS = [
        'o',  # 圆形
        's',  # 正方形
        '^',  # 上三角
        'D',  # 菱形
        'v',  # 下三角
        'p',  # 五边形
        '*',  # 星形
        'h',  # 六边形
    ]

    # 线宽
    LINEWIDTH_THIN = 1.0
    LINEWIDTH_NORMAL = 1.5
    LINEWIDTH_THICK = 2.0
    LINEWIDTH_BOLD = 2.5

    # 标记大小
    MARKERSIZE_SMALL = 4
    MARKERSIZE_NORMAL = 6
    MARKERSIZE_LARGE = 8

    @classmethod
    def get_line_style(cls, idx: int) -> Tuple[str, str, float]:
        """
        获取线条样式组合
        返回: (颜色, 线型, 线宽)
        """
        colors = JournalColors.get_main_palette(8)
        color = colors[idx % len(colors)]
        linestyle = cls.LINE_STYLES[idx % len(cls.LINE_STYLES)]
        linewidth = cls.LINEWIDTH_NORMAL
        return color, linestyle, linewidth

    @classmethod
    def get_marker_style(cls, idx: int) -> Tuple[str, str, float, float]:
        """
        获取标记样式组合
        返回: (颜色, 标记, 大小, 边框宽度)
        """
        colors = JournalColors.get_main_palette(8)
        color = colors[idx % len(colors)]
        marker = cls.MARKERS[idx % len(cls.MARKERS)]
        size = cls.MARKERSIZE_NORMAL
        edgewidth = 0.8
        return color, marker, size, edgewidth


# =============================================================================
# 期刊风格设置
# =============================================================================

def setup_journal_style():
    """配置Nature/Science期刊风格的matplotlib参数"""

    # 使用Helvetica/Arial字体家族（期刊标准）
    plt.rcParams.update({
        # 字体设置 - 期刊标准
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7,

        # 坐标轴
        'axes.linewidth': 0.8,
        'axes.edgecolor': JournalColors.CHARCOAL,
        'axes.labelcolor': JournalColors.CHARCOAL,
        'axes.facecolor': 'white',
        'axes.grid': True,
        'axes.axisbelow': True,

        # 刻度
        'xtick.major.width': 0.8,
        'xtick.minor.width': 0.6,
        'ytick.major.width': 0.8,
        'ytick.minor.width': 0.6,
        'xtick.major.size': 3.5,
        'xtick.minor.size': 2,
        'ytick.major.size': 3.5,
        'ytick.minor.size': 2,
        'xtick.color': JournalColors.CHARCOAL,
        'ytick.color': JournalColors.CHARCOAL,

        # 网格
        'grid.color': JournalColors.GRID,
        'grid.linestyle': '-',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.4,

        # 线条
        'lines.linewidth': JournalStyles.LINEWIDTH_NORMAL,
        'lines.markersize': JournalStyles.MARKERSIZE_NORMAL,
        'lines.markeredgewidth': 0.8,

        # 图例
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.fancybox': False,
        'legend.edgecolor': JournalColors.GRID,
        'legend.borderpad': 0.4,
        'legend.labelspacing': 0.4,
        'legend.handlelength': 1.8,
        'legend.handleheight': 0.7,

        # 图形
        'figure.facecolor': 'white',
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'savefig.facecolor': 'white',

        # PDF输出优化
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def mm_to_inch(mm: float) -> float:
    """毫米转英寸"""
    return mm / 25.4


def create_single_panel(width_mm: float = 86, height_mm: Optional[float] = None,
                        dpi: int = 300) -> Tuple[mpl.figure.Figure, plt.Axes]:
    """
    创建单栏期刊图形（Nature单栏标准：86mm）

    Args:
        width_mm: 宽度（毫米），Nature单栏=86mm, 双栏=178mm
        height_mm: 高度（毫米），默认为宽度（正方形）
        dpi: 分辨率
    """
    if height_mm is None:
        height_mm = width_mm  # 默认正方形

    fig, ax = plt.subplots(
        figsize=(mm_to_inch(width_mm), mm_to_inch(height_mm)),
        dpi=dpi
    )

    # 优化边距（期刊标准）
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)

    return fig, ax


def style_axis(ax: plt.Axes, show_top_right: bool = False):
    """
    应用期刊风格到坐标轴

    Args:
        ax: matplotlib轴对象
        show_top_right: 是否显示顶部和右侧边框
    """
    if not show_top_right:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    else:
        ax.spines['top'].set_linewidth(0.8)
        ax.spines['right'].set_linewidth(0.8)

    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)

    # 设置刻度参数
    ax.tick_params(
        axis='both',
        which='major',
        width=0.8,
        length=3.5,
        direction='out'
    )

    # 网格
    ax.grid(True, which='major', alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)


def save_figure(fig: mpl.figure.Figure, save_dir: str,
                filename: str, formats: List[str] = ['png', 'pdf']):
    """
    保存图形为多种格式

    Args:
        fig: matplotlib图形对象
        save_dir: 保存目录
        filename: 文件名（不含扩展名）
        formats: 保存格式列表
    """
    base_name = Path(filename).stem

    for fmt in formats:
        save_path = Path(save_dir) / f"{base_name}.{fmt}"
        fig.savefig(
            save_path,
            dpi=300 if fmt == 'png' else None,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none',
            pad_inches=0.05,
            transparent=False
        )
        print(f"  ✓ Saved: {save_path}")

    plt.close(fig)


def smooth_curve(y: np.ndarray, window: int = 5) -> np.ndarray:
    """
    平滑曲线（移动平均）

    Args:
        y: 数据
        window: 窗口大小
    """
    if window <= 1:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode='same')


# =============================================================================
# 期刊级可视化器
# =============================================================================

class JournalVisualizer:
    """期刊级别的可视化工具"""

    def __init__(self, save_dir: str, dpi: int = 300,
                 width_mm: float = 86, smooth_window: int = 1):
        """
        Args:
            save_dir: 输出目录
            dpi: 分辨率
            width_mm: 图形宽度(mm)，Nature单栏=86mm
            smooth_window: 平滑窗口大小
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.width_mm = width_mm
        self.smooth_window = smooth_window

        setup_journal_style()
        print(f"[Visualizer] Initialized with journal style")
        print(f"[Visualizer] Output directory: {self.save_dir}")

    # =========================================================================
    # 训练过程可视化
    # =========================================================================

    def plot_training_curves(self, log_file: str,
                             filename: str = "training_curves"):
        """
        训练和验证曲线（损失+准确率）
        """
        try:
            df = pd.read_csv(log_file)
            required_cols = {'epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc'}
            if not required_cols.issubset(df.columns):
                print(f"[Warning] Missing columns in {log_file}")
                return
        except Exception as e:
            print(f"[Warning] Cannot read {log_file}: {e}")
            return

        print(f"[Plot] Training curves")

        # 创建双子图
        fig, (ax1, ax2) = plt.subplots(
            2, 1,
            figsize=(mm_to_inch(self.width_mm), mm_to_inch(self.width_mm * 1.6)),
            dpi=self.dpi
        )
        plt.subplots_adjust(hspace=0.3, left=0.15, right=0.95, top=0.96, bottom=0.08)

        epochs = df['epoch'].values

        # 损失曲线
        train_loss = smooth_curve(df['train_loss'].values, self.smooth_window)
        val_loss = smooth_curve(df['val_loss'].values, self.smooth_window)

        ax1.plot(epochs, train_loss,
                 color=JournalColors.DEEP_BLUE,
                 linestyle='-',
                 linewidth=JournalStyles.LINEWIDTH_THICK,
                 marker='o',
                 markersize=3,
                 markevery=max(1, len(epochs) // 10),
                 markerfacecolor='white',
                 markeredgewidth=1.5,
                 label='Training')

        ax1.plot(epochs, val_loss,
                 color=JournalColors.VIBRANT_ORANGE,
                 linestyle='--',
                 linewidth=JournalStyles.LINEWIDTH_THICK,
                 marker='s',
                 markersize=3,
                 markevery=max(1, len(epochs) // 10),
                 markerfacecolor='white',
                 markeredgewidth=1.5,
                 label='Validation')

        ax1.set_xlabel('Epoch', fontweight='bold')
        ax1.set_ylabel('Loss', fontweight='bold')
        ax1.legend(loc='best', framealpha=0.95)
        style_axis(ax1)
        ax1.text(0.02, 0.98, 'A', transform=ax1.transAxes,
                 fontsize=12, fontweight='bold', va='top')

        # 准确率曲线
        train_acc = smooth_curve(df['train_acc'].values, self.smooth_window)
        val_acc = smooth_curve(df['val_acc'].values, self.smooth_window)

        ax2.plot(epochs, train_acc,
                 color=JournalColors.DEEP_BLUE,
                 linestyle='-',
                 linewidth=JournalStyles.LINEWIDTH_THICK,
                 marker='o',
                 markersize=3,
                 markevery=max(1, len(epochs) // 10),
                 markerfacecolor='white',
                 markeredgewidth=1.5,
                 label='Training')

        ax2.plot(epochs, val_acc,
                 color=JournalColors.VIBRANT_ORANGE,
                 linestyle='--',
                 linewidth=JournalStyles.LINEWIDTH_THICK,
                 marker='s',
                 markersize=3,
                 markevery=max(1, len(epochs) // 10),
                 markerfacecolor='white',
                 markeredgewidth=1.5,
                 label='Validation')

        ax2.set_xlabel('Epoch', fontweight='bold')
        ax2.set_ylabel('Accuracy (%)', fontweight='bold')
        ax2.set_ylim([0, 105])
        ax2.legend(loc='best', framealpha=0.95)
        style_axis(ax2)
        ax2.text(0.02, 0.98, 'B', transform=ax2.transAxes,
                 fontsize=12, fontweight='bold', va='top')

        save_figure(fig, self.save_dir, filename)

    def plot_learning_rate(self, log_file: str,
                           filename: str = "learning_rate"):
        """学习率调度可视化"""
        try:
            df = pd.read_csv(log_file)
            if 'lr' not in df.columns:
                return
        except Exception:
            return

        print(f"[Plot] Learning rate schedule")

        fig, ax = create_single_panel(self.width_mm, dpi=self.dpi)

        ax.plot(df['epoch'], df['lr'],
                color=JournalColors.FOREST_GREEN,
                linestyle='-',
                linewidth=JournalStyles.LINEWIDTH_THICK,
                marker='D',
                markersize=3,
                markevery=max(1, len(df) // 15),
                markerfacecolor='white',
                markeredgewidth=1.2)

        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Learning Rate', fontweight='bold')
        ax.set_yscale('log')
        style_axis(ax)

        save_figure(fig, self.save_dir, filename)

    # =========================================================================
    # 数据分布可视化
    # =========================================================================

    def plot_class_distribution(self, class_counts: np.ndarray,
                                class_names: Optional[List[str]] = None,
                                class_groups: Optional[Dict[str, List[int]]] = None,
                                imbalance_ratio: Optional[float] = None,
                                filename: str = "class_distribution"):
        """
        类别分布柱状图（按组着色）

        Args:
            class_counts: 每类样本数
            class_names: 类别名称
            class_groups: 类别分组 {'majority': [0,1], 'medium': [2,3], 'minority': [4,5]}
            imbalance_ratio: 预设的不平衡度，如果为None则从数据计算
            filename: 保存文件名
        """
        print(f"[Plot] Class distribution (group-colored)")

        n_classes = len(class_counts)
        labels = class_names if class_names else [f"C{i}" for i in range(n_classes)]

        fig, ax = create_single_panel(self.width_mm, dpi=self.dpi)

        # 根据分组分配颜色
        # 根据分组分配颜色
        if class_groups is None:
            # 与训练/评估一致的分组逻辑：
            counts = np.asarray(class_counts).astype(int)
            if counts.max() >= 100:
                many_thresh = 100
                few_thresh = 20
                many = np.where(counts >= many_thresh)[0]
                few = np.where(counts <= few_thresh)[0]
            else:
                q_low, q_high = 1 / 3, 2 / 3
                lo = np.quantile(counts, q_low);
                hi = np.quantile(counts, q_high)
                many = np.where(counts >= hi)[0];
                few = np.where(counts <= lo)[0]
            medium = np.setdiff1d(np.arange(n_classes), np.concatenate([many, few]))
            class_groups = {
                'majority': many.tolist(),
                'medium': medium.tolist(),
                'minority': few.tolist()
            }

        # 为每个类别分配组颜色
        group_colors = {
            'majority': JournalColors.DEEP_BLUE,  # 深蓝 - 多数类
            'medium': JournalColors.FOREST_GREEN,  # 森林绿 - 中等类
            'minority': JournalColors.VIBRANT_ORANGE  # 活力橙 - 少数类
        }

        colors = []
        for i in range(n_classes):
            assigned_color = JournalColors.STEEL_GRAY  # 默认颜色
            for group_name, class_list in class_groups.items():
                if i in class_list:
                    assigned_color = group_colors.get(group_name, JournalColors.STEEL_GRAY)
                    break
            colors.append(assigned_color)

        # 绘制柱状图
        bars = ax.bar(range(n_classes), class_counts,
                      color=colors,
                      edgecolor=JournalColors.CHARCOAL,
                      linewidth=0.8,
                      alpha=0.85,
                      width=0.7)

        ax.set_xlabel('Class', fontweight='bold')
        ax.set_ylabel('Sample Count', fontweight='bold')
        ax.set_xticks(range(n_classes))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        style_axis(ax)

        # 添加图例（只显示存在的组）
        from matplotlib.patches import Patch
        legend_elements = []
        for group_name in ['majority', 'medium', 'minority']:
            if class_groups.get(group_name):
                legend_elements.append(
                    Patch(facecolor=group_colors[group_name],
                          edgecolor=JournalColors.CHARCOAL,
                          linewidth=0.8,
                          alpha=0.85,
                          label=group_name.capitalize())
                )

        if legend_elements:
            ax.legend(handles=legend_elements,
                      loc='upper right',
                      framealpha=0.95,
                      fontsize=7)

        # 添加不平衡比（使用配置的值或计算值）
        if imbalance_ratio is None:
            imbalance_ratio = float(np.max(class_counts)) / float(np.min(class_counts))

        # ax.text(0.02, 0.98, f'IR = {imbalance_ratio:.1f}',
        #         transform=ax.transAxes,
        #         ha='left', va='top',
        #         fontsize=8,
        #         fontweight='bold',
        #         bbox=dict(boxstyle='round,pad=0.5',
        #                   facecolor='white',
        #                   edgecolor=JournalColors.GRID,
        #                   linewidth=1))

        save_figure(fig, self.save_dir, filename)

    # =========================================================================
    # 特征空间可视化
    # =========================================================================

    def plot_tsne_2d(self, features: np.ndarray, labels: np.ndarray,
                     filename: str = "tsne_2d",
                     perplexity: int = 30, n_iter: int = 1000,
                     class_names: Optional[List[str]] = None):
        """t-SNE 2D投影"""
        print(f"[Plot] t-SNE 2D projection (n={len(features)})")

        # 计算t-SNE
        try:
            tsne = TSNE(n_components=2, perplexity=perplexity,
                        max_iter=n_iter, random_state=42, verbose=0)
        except TypeError:
            tsne = TSNE(n_components=2, perplexity=perplexity,
                        n_iter=n_iter, random_state=42, verbose=0)

        embedding = tsne.fit_transform(features)

        fig, ax = create_single_panel(self.width_mm, dpi=self.dpi)

        unique_labels = np.unique(labels)
        colors = JournalColors.get_main_palette(len(unique_labels))
        markers = JournalStyles.MARKERS

        for idx, label in enumerate(unique_labels):
            mask = labels == label
            name = class_names[label] if class_names else f"Class {label}"

            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=[colors[idx]],
                       marker=markers[idx % len(markers)],
                       s=25,
                       alpha=0.7,
                       edgecolors='white',
                       linewidths=0.5,
                       label=name,
                       rasterized=True)

        ax.set_xlabel('t-SNE 1', fontweight='bold')
        ax.set_ylabel('t-SNE 2', fontweight='bold')
        style_axis(ax, show_top_right=True)

        if len(unique_labels) <= 10:
            ax.legend(loc='best', ncol=2, framealpha=0.95,
                      markerscale=1.2)

        save_figure(fig, self.save_dir, filename)

    # =========================================================================
    # 性能评估可视化
    # =========================================================================

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                              class_names: Optional[List[str]] = None,
                              filename: str = "confusion_matrix",
                              normalize: bool = True):
        """混淆矩阵热图"""
        print(f"[Plot] Confusion matrix")

        cm = confusion_matrix(y_true, y_pred)
        n_classes = cm.shape[0]
        labels = class_names if class_names else [f"{i}" for i in range(n_classes)]

        if normalize:
            cm_display = cm.astype('float') / cm.sum(axis=1, keepdims=True).clip(min=1e-10)
        else:
            cm_display = cm.astype('float')

        fig, ax = create_single_panel(self.width_mm, self.width_mm, dpi=self.dpi)

        # 使用定制色图
        from matplotlib.colors import LinearSegmentedColormap
        colors_list = ['#FFFFFF', JournalColors.LIGHT_BLUE, JournalColors.DEEP_BLUE]
        cmap = LinearSegmentedColormap.from_list('custom', colors_list)

        im = ax.imshow(cm_display, cmap=cmap, aspect='auto',
                       vmin=0, vmax=1 if normalize else cm.max())

        # 色条
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label('Proportion' if normalize else 'Count',
                       rotation=270, labelpad=15, fontsize=8, fontweight='bold')

        # 设置刻度
        ax.set_xticks(np.arange(n_classes))
        ax.set_yticks(np.arange(n_classes))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
        ax.set_xlabel('Predicted Label', fontweight='bold')
        ax.set_ylabel('True Label', fontweight='bold')

        # 添加文本标注（智能显示）
        threshold = 0.5 if normalize else cm.max() / 2
        for i in range(n_classes):
            for j in range(n_classes):
                value = cm_display[i, j]
                if i == j or value > 0.1:  # 只标注对角线和显著值
                    if normalize:
                        text = f'{value:.2f}' if value > 0.01 else ''
                    else:
                        text = f'{int(cm[i, j])}' if cm[i, j] > 0 else ''

                    if text:
                        color = 'white' if value > threshold else JournalColors.CHARCOAL
                        fontsize = 6 if n_classes > 10 else 7
                        ax.text(j, i, text, ha='center', va='center',
                                color=color, fontsize=fontsize, fontweight='bold')

        style_axis(ax, show_top_right=True)
        ax.grid(False)

        save_figure(fig, self.save_dir, filename)

    def plot_per_class_metrics(self, class_metrics: Dict[int, float],
                               class_names: Optional[List[str]] = None,
                               metric_name: str = 'Accuracy',
                               filename: str = "per_class_metrics"):
        """每类性能指标"""
        print(f"[Plot] Per-class {metric_name}")

        classes = sorted(class_metrics.keys())
        values = [class_metrics[c] for c in classes]
        labels = class_names if class_names else [f"C{c}" for c in classes]

        fig, ax = create_single_panel(self.width_mm, dpi=self.dpi)

        # 颜色编码：高于平均用深色，低于平均用浅色
        mean_val = np.mean(values)
        colors = [JournalColors.DEEP_BLUE if v >= mean_val
                  else JournalColors.LIGHT_BLUE for v in values]

        bars = ax.bar(range(len(classes)), values,
                      color=colors,
                      edgecolor=JournalColors.CHARCOAL,
                      linewidth=0.8,
                      alpha=0.9,
                      width=0.7)

        # 平均线
        ax.axhline(mean_val, color=JournalColors.CRIMSON,
                   linestyle='--', linewidth=1.5,
                   label=f'Mean = {mean_val:.1f}%', zorder=10)

        ax.set_xlabel('Class', fontweight='bold')
        ax.set_ylabel(f'{metric_name} (%)', fontweight='bold')
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylim([0, 105])
        ax.legend(loc='lower right', framealpha=0.95)
        style_axis(ax)

        save_figure(fig, self.save_dir, filename)

    def plot_sample_vs_performance(self, per_class_metrics: Dict,
                                   class_counts: np.ndarray,
                                   metric_name: str = 'recall',
                                   filename: str = "sample_vs_performance"):
        """样本数量 vs 性能关系"""
        print(f"[Plot] Sample count vs {metric_name}")

        classes = sorted([int(k) for k in per_class_metrics.keys()])
        metric_values = [per_class_metrics[str(c)][metric_name] for c in classes]
        counts = [class_counts[c] for c in classes]

        fig, ax = create_single_panel(self.width_mm, dpi=self.dpi)

        # 散点图
        ax.scatter(counts, metric_values,
                   s=60,
                   c=JournalColors.DEEP_BLUE,
                   marker='o',
                   alpha=0.7,
                   edgecolors=JournalColors.CHARCOAL,
                   linewidths=1.2,
                   zorder=3)

        # 拟合趋势线
        log_counts = np.log10(np.clip(counts, 1, None))
        z = np.polyfit(log_counts, metric_values, 1)
        p = np.poly1d(z)
        x_fit = np.logspace(np.log10(min(counts)), np.log10(max(counts)), 100)
        ax.plot(x_fit, p(np.log10(x_fit)),
                color=JournalColors.CRIMSON,
                linestyle='--',
                linewidth=2,
                alpha=0.8,
                label=f'Trend: R²={np.corrcoef(log_counts, metric_values)[0, 1] ** 2:.3f}',
                zorder=2)

        ax.set_xlabel('Sample Count (log scale)', fontweight='bold')
        ax.set_ylabel(f'{metric_name.capitalize()} (%)', fontweight='bold')
        ax.set_xscale('log')
        ax.set_ylim([0, 105])
        ax.legend(loc='lower right', framealpha=0.95)
        style_axis(ax)

        save_figure(fig, self.save_dir, filename)

    def plot_confidence_distribution(self, y_true: np.ndarray,
                                     y_pred: np.ndarray,
                                     probs: np.ndarray,
                                     filename: str = "confidence_dist"):
        """预测置信度分布"""
        print(f"[Plot] Confidence distribution")

        max_probs = np.max(probs, axis=1)
        correct_mask = (y_true == y_pred)

        fig, ax = create_single_panel(self.width_mm, dpi=self.dpi)

        bins = np.linspace(0, 1, 30)

        # 正确预测
        ax.hist(max_probs[correct_mask], bins=bins,
                alpha=0.7,
                color=JournalColors.FOREST_GREEN,
                edgecolor='white',
                linewidth=0.5,
                label=f'Correct ({correct_mask.sum()})',
                density=True)

        # 错误预测
        ax.hist(max_probs[~correct_mask], bins=bins,
                alpha=0.7,
                color=JournalColors.VIBRANT_ORANGE,
                edgecolor='white',
                linewidth=0.5,
                label=f'Incorrect ({(~correct_mask).sum()})',
                density=True)

        ax.set_xlabel('Prediction Confidence', fontweight='bold')
        ax.set_ylabel('Density', fontweight='bold')
        ax.set_xlim([0, 1])
        ax.legend(loc='upper left', framealpha=0.95)
        style_axis(ax)

        save_figure(fig, self.save_dir, filename)

    def plot_top_confusions(self, y_true: np.ndarray, y_pred: np.ndarray,
                            class_names: Optional[List[str]] = None,
                            top_k: int = 10,
                            filename: str = "top_confusions"):
        """Top混淆对"""
        print(f"[Plot] Top {top_k} confusion pairs")

        cm = confusion_matrix(y_true, y_pred)
        n_classes = cm.shape[0]
        labels = class_names if class_names else [f"C{i}" for i in range(n_classes)]

        # 收集混淆对
        confusions = []
        for i in range(n_classes):
            for j in range(n_classes):
                if i != j and cm[i, j] > 0:
                    confusions.append((i, j, cm[i, j]))

        confusions.sort(key=lambda x: x[2], reverse=True)
        top_confusions = confusions[:min(top_k, len(confusions))]

        if not top_confusions:
            return

        fig, ax = create_single_panel(self.width_mm * 1.3, dpi=self.dpi)

        pair_labels = [f"{labels[i]} → {labels[j]}"
                       for i, j, _ in top_confusions]
        counts = [count for _, _, count in top_confusions]

        y_pos = np.arange(len(pair_labels))
        colors_grad = JournalColors.get_sequential_blues(len(pair_labels))[::-1]

        bars = ax.barh(y_pos, counts, color=colors_grad,
                       edgecolor=JournalColors.CHARCOAL,
                       linewidth=0.8, alpha=0.9)

        # 添加数值标签
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(count + max(counts) * 0.02, i, f'{int(count)}',
                    va='center', ha='left', fontsize=7, fontweight='bold')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(pair_labels, fontsize=7.5)
        ax.set_xlabel('Confusion Count', fontweight='bold')
        ax.invert_yaxis()
        style_axis(ax)

        save_figure(fig, self.save_dir, filename)

    def plot_group_performance(self, group_metrics: Dict,
                               filename: str = "group_performance"):
        """组别性能对比"""
        if not group_metrics:
            return

        print(f"[Plot] Group-wise performance")

        groups, accuracies, f1_scores = [], [], []
        for group_name in ['majority', 'medium', 'minority']:
            if group_name in group_metrics and group_metrics[group_name]:
                groups.append(group_name.capitalize())
                accuracies.append(group_metrics[group_name]['accuracy'])
                f1_scores.append(group_metrics[group_name]['f1'])

        if not groups:
            return

        fig, ax = create_single_panel(self.width_mm, dpi=self.dpi)

        y_pos = np.arange(len(groups))
        width = 0.35

        ax.barh(y_pos - width / 2, accuracies, width,
                label='Accuracy',
                color=JournalColors.DEEP_BLUE,
                edgecolor=JournalColors.CHARCOAL,
                linewidth=0.8,
                alpha=0.9)

        ax.barh(y_pos + width / 2, f1_scores, width,
                label='F1-Score',
                color=JournalColors.VIBRANT_ORANGE,
                edgecolor=JournalColors.CHARCOAL,
                linewidth=0.8,
                alpha=0.9)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(groups)
        ax.set_xlabel('Performance (%)', fontweight='bold')
        ax.set_xlim([0, 100])
        ax.legend(loc='lower right', framealpha=0.95)
        style_axis(ax)

        save_figure(fig, self.save_dir, filename)


# =============================================================================
# 特征提取辅助函数
# =============================================================================

def extract_features_and_predictions(model: nn.Module,
                                     loader: DataLoader,
                                     device: torch.device,
                                     max_samples: Optional[int] = None) -> Tuple:
    """
    从模型提取特征、标签和预测

    Returns:
        features, labels, predictions, probabilities
    """
    model.eval()
    all_features, all_labels, all_preds, all_probs = [], [], [], []

    print(f"[Extract] Extracting features and predictions...")
    sample_count = 0

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # 前向传播
            output = model(x)
            if isinstance(output, tuple) and len(output) >= 2:
                logits, features = output[0], output[1]
            else:
                logits = output
                features = logits

            # 预测和概率
            probs = torch.softmax(logits, dim=1)
            _, preds = logits.max(1)

            # 收集
            all_features.append(features.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

            sample_count += x.size(0)
            if max_samples and sample_count >= max_samples:
                break

            if (batch_idx + 1) % 20 == 0:
                print(f"  Processed: {sample_count} samples")

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    predictions = np.concatenate(all_preds, axis=0)
    probabilities = np.concatenate(all_probs, axis=0)

    print(f"[Extract] Completed: {len(features)} samples")

    return features, labels, predictions, probabilities


# =============================================================================
# 主可视化函数
# =============================================================================

def visualize_all_results(model: nn.Module,
                                 test_loader: DataLoader,
                                 device: torch.device,
                                 save_dir: str,
                                 logs_dir: str,
                                 test_results: Dict,
                                 class_counts: np.ndarray,
                                 config: Dict,
                                 class_names: Optional[List[str]] = None):
    """
    完整实验结果可视化流程

    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 计算设备
        save_dir: 保存目录
        logs_dir: 日志目录
        test_results: 测试结果字典
        class_counts: 类别样本数
        config: 配置字典
        class_names: 类别名称列表
    """
    viz_dir = Path(save_dir) / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("JOURNAL-STYLE VISUALIZATION")
    print("=" * 70)

    # 创建可视化器
    viz = JournalVisualizer(
        save_dir=str(viz_dir),
        dpi=config.get('dpi', 300),
        width_mm=config.get('width_mm', 86),
        smooth_window=config.get('smooth_window', 1)
    )

    # 1. 训练曲线
    log_file = Path(logs_dir) / "training.log"
    if log_file.exists():
        if config.get('plot_training_curves', True):
            viz.plot_training_curves(str(log_file))
        if config.get('plot_learning_rate', True):
            viz.plot_learning_rate(str(log_file))

    # 2. 提取特征和预测
    print(f"\n{'=' * 70}")
    max_samples = config.get('max_samples', None)
    features, labels, predictions, probs = extract_features_and_predictions(
        model, test_loader, device, max_samples
    )

    # 3. 生成各类图表
    print(f"\n{'=' * 70}")
    print("Generating visualizations...")
    print(f"{'=' * 70}\n")

    if config.get('plot_class_distribution', True):


        # 获取配置的不平衡度（优先级：config > dataset_info > 计算值）
        imbalance_ratio = None
        if 'imbalance_ratio' in config:
            imbalance_ratio = config['imbalance_ratio']
        elif 'dataset' in test_results and 'imbalance_ratio' in test_results['dataset']:
            imbalance_ratio = test_results['dataset']['imbalance_ratio']
        elif 'dataset_info' in config and 'imbalance_ratio' in config['dataset_info']:
            imbalance_ratio = config['dataset_info']['imbalance_ratio']

        viz.plot_class_distribution(
            class_counts,
            class_names,
            class_groups=None,
            imbalance_ratio=imbalance_ratio
        )

    if config.get('plot_tsne_2d', True):
        viz.plot_tsne_2d(features, labels,
                         perplexity=config.get('tsne_perplexity', 30),
                         n_iter=config.get('tsne_n_iter', 1000),
                         class_names=class_names)

    if config.get('plot_confusion_matrix', True):
        viz.plot_confusion_matrix(labels, predictions,
                                  class_names=class_names,
                                  normalize=config.get('cm_normalize', True))

    if config.get('plot_per_class_metrics', True):
        per_class_acc = {int(cid): metrics['recall']
                         for cid, metrics in test_results['per_class'].items()}
        viz.plot_per_class_metrics(per_class_acc, class_names,
                                   metric_name='Recall',
                                   filename='per_class_recall')

    if config.get('plot_group_performance', True):
        viz.plot_group_performance(test_results.get('group_wise', {}))

    if config.get('plot_sample_vs_performance', True):
        viz.plot_sample_vs_performance(test_results['per_class'],
                                       class_counts,
                                       metric_name='recall',
                                       filename='sample_vs_recall')
        viz.plot_sample_vs_performance(test_results['per_class'],
                                       class_counts,
                                       metric_name='f1',
                                       filename='sample_vs_f1')

    if config.get('plot_confidence_dist', True):
        viz.plot_confidence_distribution(labels, predictions, probs)

    if config.get('plot_top_confusions', True):
        viz.plot_top_confusions(labels, predictions,
                                class_names=class_names,
                                top_k=config.get('top_confusions_k', 10))

    print(f"\n{'=' * 70}")
    print(f"✓ All visualizations saved to: {viz_dir}")
    print(f"{'=' * 70}\n")




# =============================================================================
# 独立运行 - 从实验目录重新生成
# =============================================================================

def regenerate_from_experiment(exp_dir: str,
                               device: str = 'auto',
                               config_override: Optional[Dict] = None):
    """
    从已有实验目录重新生成所有可视化

    Args:
        exp_dir: 实验目录路径
        device: 计算设备 ('auto', 'cuda', 'cpu')
        config_override: 覆盖配置
    """
    import json

    exp_path = Path(exp_dir)

    print("\n" + "=" * 70)
    print("REGENERATING VISUALIZATIONS FROM EXPERIMENT")
    print("=" * 70)
    print(f"Experiment directory: {exp_path}\n")

    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_path}")

    # 加载配置和结果
    config_file = exp_path / "config.json"
    results_file = exp_path / "results" / "results.json"

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    with open(config_file, 'r') as f:
        config = json.load(f)
    with open(results_file, 'r') as f:
        results = json.load(f)

    print(f"✓ Loaded config and results\n")

    # 设置设备
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    print(f"✓ Device: {device}\n")

    # 查找checkpoint
    ckpt_paths = [
        exp_path / "checkpoints_stage2" / "best.pth",
        exp_path / "checkpoints" / "best.pth",
    ]

    checkpoint_path = None
    for path in ckpt_paths:
        if path.exists():
            checkpoint_path = path
            break

    if checkpoint_path is None:
        raise FileNotFoundError("No checkpoint found")

    print(f"✓ Found checkpoint: {checkpoint_path.name}\n")

    # 加载模型（需要导入相关模块）
    try:
        from models import create_model
        from data_utils import ADSBSignalDataset
    except ImportError:
        raise ImportError("Required modules not found. Ensure models.py and data_utils.py are available.")

    num_classes = results['dataset']['num_classes']
    model_config = config['model']

    model = create_model(
        model_config['name'],
        num_classes=num_classes,
        dropout_rate=model_config.get('dropout', 0.1),
        use_attention=model_config.get('use_attention', False),
        norm_kind=model_config.get('norm_kind', 'auto')
    )

    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # 处理 DataParallel
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k[7:] if k.startswith('module.') else k
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded\n")

    # 加载测试数据
    test_data_path = config['data']['path_test']
    dataset = ADSBSignalDataset(
        path=test_data_path,
        target_length=config['data'].get('target_length'),
        normalize=config['data'].get('normalize', True),
        seed=config.get('seed', 2027)
    )

    test_loader = DataLoader(
        dataset,
        batch_size=config['data'].get('batch_size', 256),
        shuffle=False,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=(device.type == 'cuda'),
        drop_last=False
    )

    print(f"✓ Test data loaded: {len(dataset)} samples\n")

    # 合并配置
    viz_config = config.get('visualization', {})
    defaults = {
        'dpi': 600,
        'width_mm': 86,
        'smooth_window': 1,
        'plot_training_curves': True,
        'plot_learning_rate': True,
        'plot_class_distribution': True,
        'plot_tsne_2d': True,
        'plot_confusion_matrix': True,
        'cm_normalize': True,
        'plot_per_class_metrics': True,
        'plot_group_performance': True,
        'plot_sample_vs_performance': True,
        'plot_confidence_dist': True,
        'plot_top_confusions': True,
        'top_confusions_k': 10,
    }

    for k, v in defaults.items():
        viz_config.setdefault(k, v)

    if config_override:
        viz_config.update(config_override)

    class_names = viz_config.get('class_names')

    # 生成可视化
    visualize_all_results(
        model=model,
        test_loader=test_loader,
        device=device,
        save_dir=str(exp_path / "results"),
        logs_dir=str(exp_path / "logs"),
        test_results=results['test_results'],
        class_counts=np.array(results['dataset']['class_counts']),
        config=viz_config,
        class_names=class_names
    )

    print("✓ Regeneration complete!\n")


def find_latest_experiment(base_dir: str = "experiments") -> Optional[str]:
    """查找最新的实验目录"""
    base_path = Path(base_dir)
    if not base_path.exists():
        return None

    exp_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    if not exp_dirs:
        return None

    latest = max(exp_dirs, key=lambda x: x.stat().st_mtime)
    return str(latest)


# =============================================================================
# CLI 入口
# =============================================================================

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Journal-Style Scientific Visualization for ML Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python journal_viz.py --exp_dir experiments/exp_20240101_120000
  python journal_viz.py --device cuda --dpi 600
  python journal_viz.py --max_samples 5000 --width_mm 86
        """
    )

    parser.add_argument('--exp_dir', type=str, default=None,
                        help='Experiment directory (default: latest)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Computation device')
    parser.add_argument('--dpi', type=int, default=None,
                        help='Figure resolution (default: 300)')
    parser.add_argument('--width_mm', type=float, default=None,
                        help='Figure width in mm (default: 86, Nature single column)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum samples for feature extraction')
    parser.add_argument('--smooth_window', type=int, default=None,
                        help='Smoothing window for curves')

    args = parser.parse_args()

    # 确定实验目录
    exp_dir = args.exp_dir
    if exp_dir is None:
        print("Searching for latest experiment...")
        exp_dir = find_latest_experiment()
        if exp_dir is None:
            print("ERROR: No experiment directory found")
            sys.exit(1)
        print(f"✓ Found latest experiment: {exp_dir}\n")

    # 准备配置覆盖
    config_override = {}
    if args.dpi is not None:
        config_override['dpi'] = args.dpi
    if args.width_mm is not None:
        config_override['width_mm'] = args.width_mm
    if args.max_samples is not None:
        config_override['max_samples'] = args.max_samples
    if args.smooth_window is not None:
        config_override['smooth_window'] = args.smooth_window

    # 运行
    try:
        regenerate_from_experiment(
            exp_dir=exp_dir,
            device=args.device,
            config_override=config_override if config_override else None
        )
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)