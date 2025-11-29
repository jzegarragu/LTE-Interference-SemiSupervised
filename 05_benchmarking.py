"""
Resource Benchmarking Script
----------------------------
Measures model complexity (Parameter count, Memory footprint) and 
inference speed (FPS) to analyze the trade-off between accuracy and efficiency.
"""

import torch
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

try:
    import models
    import config
except ImportError:
    print("[ERROR] Ensure models.py and config.py are in the working directory.")
    sys.exit(1)

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output Directory (Relative path based on config)
OUTPUT_DIR = config.ARTIFACTS_DIR / "benchmark"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def measure_resources(model_name, num_classes=5):
    """
    Measures parameter count and inference time using dummy data.
    """
    print(f"[INFO] Benchmarking {model_name}...")
    
    # Initialize model (Pretraining false as we only care about architecture size)
    model = models.make_model(model_name, num_classes, resnet_use_pretrain=False).to(DEVICE)
    model.eval()

    # 1. Parameter Count
    num_params = sum(p.numel() for p in model.parameters())
    model_size_mb = num_params * 4 / (1024 * 1024)  # Assuming float32 (4 bytes)

    # 2. Inference Speed (Batch Size 32)
    dummy_input = torch.randn(32, 1, 100, 100).to(DEVICE)
    
    # Warm-up (stabilize GPU/CPU cache)
    for _ in range(10): 
        _ = model(dummy_input)
    
    # Measurement Loop
    iterations = 100
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time_per_batch = (end_time - start_time) / iterations
    fps = 32 / avg_time_per_batch

    return {
        "Model": model_name,
        "Params (M)": num_params / 1e6, # In Millions
        "Size (MB)": model_size_mb,
        "FPS": fps
    }

def plot_dual_axis(df):
    """
    Generates a publication-quality dual-axis plot (Bar + Line).
    """
    # Professional style (Clean white background)
    sns.set_style("white")
    plt.rcParams.update({'font.family': 'serif', 'font.size': 12})

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # --- AXIS 1 (Left): BARS (Parameters) ---
    color_bar = '#4c72b0' # Muted Blue
    bars = sns.barplot(
        data=df, 
        x='Model', 
        y='Params (M)', 
        ax=ax1, 
        color=color_bar, 
        alpha=0.8, 
        width=0.5
    )
    ax1.set_ylabel('Parameters (Millions)', color=color_bar, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_bar)
    ax1.set_xlabel('Architecture', fontweight='bold')
    
    # Bar Labels
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.1f M', padding=3, color=color_bar, fontweight='bold')

    # --- AXIS 2 (Right): LINE (Throughput) ---
    ax2 = ax1.twinx()
    color_line = '#c44e52' # Muted Red
    
    # Plot line with markers
    sns.lineplot(
        data=df, 
        x='Model', 
        y='FPS', 
        ax=ax2, 
        color=color_line, 
        marker='o', 
        markersize=10, 
        linewidth=3, 
        sort=False
    )
    ax2.set_ylabel('Throughput (Images / sec)', color=color_line, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_line)
    
    # Line Point Labels
    for i, point in df.iterrows():
        ax2.text(
            i, 
            point['FPS'] + (point['FPS'] * 0.05), 
            f"{point['FPS']:.0f} FPS", 
            color=color_line, 
            ha='center', 
            fontweight='bold'
        )

    # --- Title and Legend ---
    plt.title('Trade-off: Model Complexity vs. Inference Speed', fontsize=14, pad=20)
    
    # Custom Legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    
    legend_elements = [
        Patch(facecolor=color_bar, edgecolor=color_bar, alpha=0.8, label='Parameters (M)'),
        Line2D([0], [0], color=color_line, lw=3, marker='o', label='Speed (FPS)')
    ]
    
    # Place legend outside
    ax1.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.15, 0.5), frameon=False)

    plt.tight_layout()
    
    # Save Plot
    save_path = OUTPUT_DIR / "resource_comparison_paper.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Plot saved to: {save_path}")

if __name__ == "__main__":
    print(f"=== PAPER BENCHMARKING (Device: {DEVICE}) ===\n")
    
    results = []
    # Architectures to compare
    for arch in ["resnet50", "resnet18"]:
        results.append(measure_resources(arch))
    
    df = pd.DataFrame(results)
    
    # Save raw data for tables
    excel_path = OUTPUT_DIR / "resource_metrics.xlsx"
    df.to_excel(excel_path, index=False)
    print(f"[INFO] Metrics saved to: {excel_path}")
    
    # Generate visualization
    plot_dual_axis(df)
    
    print("\n[DONE] Benchmark completed.")