# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Test temporal pooling with trained videoseal model.

There are TWO independent mechanisms for reducing frame computation:

1. **step_size** (frame skipping before UNet):
   - Only every Nth frame is processed by the embedder
   - video_mode controls how watermarks propagate to skipped frames
   - Example: step_size=4 means only frames 0,4,8,12 are embedded

2. **temporal_pooling** (pooling inside UNet):
   - ALL frames enter the UNet, but frames are pooled at a specific depth
   - Reduces computation in deeper UNet layers
   - Controlled via model.embedder.unet.time_pooling* attributes

These are MUTUALLY EXCLUSIVE - when temporal_pooling is enabled,
step_size should be set to 1 (frame skipping is disabled).

Run with:
    python -m videoseal.tests.test_videoseal_temporal
"""

import copy
import time
import torch
import torchvision
import unittest
import matplotlib.pyplot as plt
from pathlib import Path

import videoseal
from videoseal.evals.metrics import bit_accuracy


def enable_temporal_pooling(model, kernel_size=2, depth=1):
    """
    Enable temporal pooling on the model's UNet.
    
    This pools frames INSIDE the UNet at a specific depth, reducing
    computation in deeper layers. When enabled, step_size should be 1.
    
    Args:
        model: Videoseal model (will be modified in place)
        kernel_size: Number of frames to pool together (e.g., 2 or 4)
        depth: UNet depth at which to apply pooling
    """
    if hasattr(model, 'embedder') and hasattr(model.embedder, 'unet'):
        unet = model.embedder.unet
        if hasattr(unet, 'time_pooling'):
            unet.time_pooling = True
            unet.time_pooling_depth = depth
            unet.temporal_pool.kernel_size = kernel_size
            unet.temporal_pool.stride = kernel_size
            # When using temporal pooling, step_size must be 1
            model.step_size = 1
            return True
    return False


class TestStepSizeFrameSkipping(unittest.TestCase):
    """
    Test step_size mechanism (frame skipping BEFORE UNet).
    
    step_size=N means only every Nth frame goes through the embedder.
    video_mode controls how watermarks propagate to skipped frames:
    - "repeat": copy watermark to adjacent frames
    - "alternate": only watermark key frames, zeros elsewhere
    - "interpolate": blend between consecutive key frames
    """

    @classmethod
    def setUpClass(cls):
        """Load model and video once."""
        cls.model = videoseal.load("pixelseal")
        cls.model.eval()
        cls.nbits = cls.model.embedder.msg_processor.nbits
        
        video_path = Path(__file__).parent.parent.parent / "assets" / "videos" / "1.mp4"
        video, _, _ = torchvision.io.read_video(str(video_path), pts_unit='sec')
        cls.video = video.permute(0, 3, 1, 2).float() / 255.0
        cls.video = cls.video[:16]  # 16 frames
        
        print(f"\n[StepSize Tests] Model: {cls.nbits} bits, Video: {cls.video.shape}")

    def _get_accuracy(self, model, video, msg):
        """Embed, detect, return mean bit accuracy."""
        with torch.no_grad():
            outputs = model.embed(video, msg, is_video=True)
            preds = model.detect(outputs["imgs_w"], is_video=True)
        pred_msgs = preds["preds"][:, 1:]
        return bit_accuracy(pred_msgs, msg).mean().item()

    def test_step_size_1_baseline(self):
        """step_size=1: watermark every frame (best quality, slowest)."""
        model = copy.deepcopy(self.model)
        model.step_size = 1
        model.video_mode = "repeat"
        model.eval()
        
        torch.manual_seed(42)
        msg = torch.randint(0, 2, (1, self.nbits)).float()
        acc = self._get_accuracy(model, self.video, msg)
        
        print(f"\n  step_size=1: {acc*100:.1f}%")
        self.assertGreater(acc, 0.95)

    def test_step_size_4_repeat(self):
        """step_size=4 with repeat: watermark frames 0,4,8,12, copy to others."""
        model = copy.deepcopy(self.model)
        model.step_size = 4
        model.video_mode = "repeat"
        model.eval()
        
        torch.manual_seed(42)
        msg = torch.randint(0, 2, (1, self.nbits)).float()
        acc = self._get_accuracy(model, self.video, msg)
        
        print(f"\n  step_size=4, repeat: {acc*100:.1f}%")
        self.assertGreater(acc, 0.80)

    def test_video_modes_comparison(self):
        """Compare video_mode options with step_size=4."""
        print("\n  Video modes (step_size=4):")
        
        torch.manual_seed(42)
        msg = torch.randint(0, 2, (1, self.nbits)).float()
        results = {}
        
        for mode in ["repeat", "alternate", "interpolate"]:
            model = copy.deepcopy(self.model)
            model.step_size = 4
            model.video_mode = mode
            model.eval()
            
            acc = self._get_accuracy(model, self.video, msg)
            results[mode] = acc
            print(f"    {mode}: {acc*100:.1f}%")
        
        # repeat should be best since watermark is on all frames
        self.assertGreater(results["repeat"], results["alternate"])


class TestTemporalPooling(unittest.TestCase):
    """
    Test temporal pooling mechanism (pooling INSIDE UNet).
    
    Temporal pooling reduces frames at a specific UNet depth:
    - All frames enter the UNet
    - At depth N, frames are pooled (e.g., 16 frames -> 8 frames)
    - Deeper layers process fewer frames (faster)
    - On upsampling, frames are expanded back
    
    When using temporal pooling, step_size MUST be 1.
    """

    @classmethod
    def setUpClass(cls):
        """Load model and video once."""
        cls.model = videoseal.load("pixelseal")
        cls.model.eval()
        cls.nbits = cls.model.embedder.msg_processor.nbits
        
        video_path = Path(__file__).parent.parent.parent / "assets" / "videos" / "1.mp4"
        video, _, _ = torchvision.io.read_video(str(video_path), pts_unit='sec')
        cls.video = video.permute(0, 3, 1, 2).float() / 255.0
        cls.video = cls.video[:16]  # 16 frames
        
        print(f"\n[Temporal Pooling Tests] Model: {cls.nbits} bits, Video: {cls.video.shape}")

    def _get_accuracy(self, model, video, msg):
        """Embed, detect, return mean bit accuracy."""
        with torch.no_grad():
            outputs = model.embed(video, msg, is_video=True)
            preds = model.detect(outputs["imgs_w"], is_video=True)
        pred_msgs = preds["preds"][:, 1:]
        return bit_accuracy(pred_msgs, msg).mean().item()

    def test_temporal_pooling_kernel_2(self):
        """Temporal pooling with kernel_size=2 (pool every 2 frames)."""
        model = copy.deepcopy(self.model)
        enabled = enable_temporal_pooling(model, kernel_size=2, depth=1)
        
        if not enabled:
            self.skipTest("Model doesn't support temporal pooling")
        
        model.eval()
        
        torch.manual_seed(42)
        msg = torch.randint(0, 2, (1, self.nbits)).float()
        acc = self._get_accuracy(model, self.video, msg)
        
        print(f"\n  temporal_pooling kernel=2: {acc*100:.1f}%")
        self.assertGreater(acc, 0.90)

    def test_temporal_pooling_kernel_4(self):
        """Temporal pooling with kernel_size=4 (pool every 4 frames)."""
        model = copy.deepcopy(self.model)
        enabled = enable_temporal_pooling(model, kernel_size=4, depth=1)
        
        if not enabled:
            self.skipTest("Model doesn't support temporal pooling")
        
        model.eval()
        
        torch.manual_seed(42)
        msg = torch.randint(0, 2, (1, self.nbits)).float()
        acc = self._get_accuracy(model, self.video, msg)
        
        print(f"\n  temporal_pooling kernel=4: {acc*100:.1f}%")
        self.assertGreater(acc, 0.80)

    def test_temporal_pooling_vs_step_size(self):
        """
        Compare temporal pooling vs step_size at equivalent reduction factor.
        
        Both reduce computation by ~4x:
        - step_size=4: skip 3/4 frames before UNet
        - temporal_pooling kernel=4: pool inside UNet
        """
        print("\n  Comparing equivalent 4x reduction:")
        
        torch.manual_seed(42)
        msg = torch.randint(0, 2, (1, self.nbits)).float()
        
        # Method 1: step_size=4 (frame skipping)
        model1 = copy.deepcopy(self.model)
        model1.step_size = 4
        model1.video_mode = "repeat"
        model1.eval()
        acc_step = self._get_accuracy(model1, self.video, msg)
        
        # Method 2: temporal_pooling kernel=4
        model2 = copy.deepcopy(self.model)
        enabled = enable_temporal_pooling(model2, kernel_size=4, depth=1)
        
        if not enabled:
            print(f"    step_size=4: {acc_step*100:.1f}%")
            print(f"    temporal_pooling: N/A (not supported)")
            self.skipTest("Model doesn't support temporal pooling")
            return
        
        model2.eval()
        acc_tp = self._get_accuracy(model2, self.video, msg)
        
        print(f"    step_size=4: {acc_step*100:.1f}%")
        print(f"    temporal_pooling kernel=4: {acc_tp*100:.1f}%")
        
        # Both should achieve reasonable accuracy
        self.assertGreater(acc_step, 0.70)
        self.assertGreater(acc_tp, 0.70)


class TestAccuracySpeedComparison(unittest.TestCase):
    """
    Compare all methods: accuracy vs speed tradeoff.
    Generates a plot showing the Pareto frontier of methods.
    """

    @classmethod
    def setUpClass(cls):
        """Load model and video once."""
        cls.model = videoseal.load("pixelseal")
        cls.model.eval()
        cls.nbits = cls.model.embedder.msg_processor.nbits
        
        video_path = Path(__file__).parent.parent.parent / "assets" / "videos" / "1.mp4"
        video, _, _ = torchvision.io.read_video(str(video_path), pts_unit='sec')
        cls.video = video.permute(0, 3, 1, 2).float() / 255.0
        cls.video = cls.video[:16]  # 16 frames
        
        print(f"\n[Accuracy vs Speed] Model: {cls.nbits} bits, Video: {cls.video.shape}")

    def _benchmark(self, model, video, msg, warmup=1, runs=3):
        """Measure accuracy and embedding time."""
        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = model.embed(video, msg, is_video=True)
        
        # Timed runs
        times = []
        for _ in range(runs):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()
            with torch.no_grad():
                outputs = model.embed(video, msg, is_video=True)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            times.append(time.time() - start)
        
        # Get accuracy
        with torch.no_grad():
            preds = model.detect(outputs["imgs_w"], is_video=True)
        pred_msgs = preds["preds"][:, 1:]
        acc = bit_accuracy(pred_msgs, msg).mean().item()
        
        avg_time = sum(times) / len(times)
        ms_per_frame = (avg_time / video.shape[0]) * 1000
        
        return acc, ms_per_frame

    def test_plot_accuracy_vs_speed(self):
        """Generate accuracy vs speed comparison plot."""
        print("\n  Benchmarking all configurations...")
        
        torch.manual_seed(42)
        msg = torch.randint(0, 2, (1, self.nbits)).float()
        
        results = []
        
        # 1. Step size variations (frame skipping)
        for step_size in [1, 2, 4, 8]:
            model = copy.deepcopy(self.model)
            model.step_size = step_size
            model.video_mode = "repeat"
            model.eval()
            
            acc, ms = self._benchmark(model, self.video, msg)
            results.append({
                "method": f"step_size={step_size}",
                "category": "step_size",
                "acc": acc,
                "ms_per_frame": ms,
                "reduction": step_size
            })
            print(f"    step_size={step_size}: {acc*100:.1f}%, {ms:.1f} ms/frame")
        
        # 2. Temporal pooling variations
        for kernel_size in [2, 4, 8]:
            model = copy.deepcopy(self.model)
            enabled = enable_temporal_pooling(model, kernel_size=kernel_size, depth=1)
            
            if not enabled:
                print(f"    temporal_pooling k={kernel_size}: SKIPPED (not supported)")
                continue
            
            model.eval()
            acc, ms = self._benchmark(model, self.video, msg)
            results.append({
                "method": f"temp_pool k={kernel_size}",
                "category": "temporal_pooling",
                "acc": acc,
                "ms_per_frame": ms,
                "reduction": kernel_size
            })
            print(f"    temporal_pooling k={kernel_size}: {acc*100:.1f}%, {ms:.1f} ms/frame")
        
        # 3. Video mode variations with step_size=4
        for mode in ["repeat", "alternate", "interpolate"]:
            model = copy.deepcopy(self.model)
            model.step_size = 4
            model.video_mode = mode
            model.eval()
            
            acc, ms = self._benchmark(model, self.video, msg)
            results.append({
                "method": f"step=4, {mode}",
                "category": "video_mode",
                "acc": acc,
                "ms_per_frame": ms,
                "reduction": 4
            })
            print(f"    step_size=4, {mode}: {acc*100:.1f}%, {ms:.1f} ms/frame")
        
        # Create the plot
        self._create_plot(results)
        
        # Verify we have results
        self.assertGreater(len(results), 0)

    def _create_plot(self, results):
        """Create accuracy vs speed comparison plot."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Color schemes
        colors = {
            "step_size": "#2ecc71",       # green
            "temporal_pooling": "#3498db", # blue  
            "video_mode": "#e74c3c"        # red
        }
        markers = {
            "step_size": "o",
            "temporal_pooling": "s",
            "video_mode": "^"
        }
        
        # ===== Plot 1: Accuracy vs Speed (scatter) =====
        ax = axes[0]
        for cat in ["step_size", "temporal_pooling", "video_mode"]:
            cat_results = [r for r in results if r["category"] == cat]
            if not cat_results:
                continue
            
            x = [r["ms_per_frame"] for r in cat_results]
            y = [r["acc"] * 100 for r in cat_results]
            labels = [r["method"] for r in cat_results]
            
            ax.scatter(x, y, c=colors[cat], marker=markers[cat], s=150, 
                      label=cat.replace("_", " ").title(), edgecolors='white', linewidths=1.5)
            
            # Add labels
            for xi, yi, label in zip(x, y, labels):
                ax.annotate(label, (xi, yi), textcoords="offset points", 
                           xytext=(5, 5), fontsize=8, alpha=0.8)
        
        ax.set_xlabel("Time (ms/frame)", fontsize=12)
        ax.set_ylabel("Bit Accuracy (%)", fontsize=12)
        ax.set_title("Accuracy vs Speed Tradeoff", fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(50, 105)
        
        # ===== Plot 2: Bar chart comparison =====
        ax = axes[1]
        
        # Sort by accuracy
        sorted_results = sorted(results, key=lambda x: x["acc"], reverse=True)
        methods = [r["method"] for r in sorted_results]
        accuracies = [r["acc"] * 100 for r in sorted_results]
        times = [r["ms_per_frame"] for r in sorted_results]
        bar_colors = [colors[r["category"]] for r in sorted_results]
        
        x_pos = range(len(methods))
        
        # Accuracy bars
        bars = ax.bar(x_pos, accuracies, color=bar_colors, alpha=0.8, edgecolor='white', linewidth=1.5)
        
        # Add time annotations on bars
        for i, (bar, t, acc) in enumerate(zip(bars, times, accuracies)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{t:.1f}ms', ha='center', va='bottom', fontsize=8, color='gray')
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                   f'{acc:.1f}%', ha='center', va='center', fontsize=9, 
                   color='white', fontweight='bold')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel("Bit Accuracy (%)", fontsize=12)
        ax.set_title("Methods Ranked by Accuracy\n(time shown above bars)", fontsize=14, fontweight='bold')
        ax.set_ylim(0, 115)
        ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5, label='90% threshold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add legend for categories
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors[cat], label=cat.replace("_", " ").title()) 
                          for cat in colors.keys()]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        plt.tight_layout()
        output_path = Path(__file__).parent / "accuracy_vs_speed_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n  Plot saved to: {output_path}")
        plt.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
