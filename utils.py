"""
Utility Functions Module
------------------------
Contains reusable helper functions for data handling, visualization, 
checkpoint management, and results logging.
"""

import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def load_png_gray(path: Path):
    """
    Loads a PNG image, converts it to grayscale, and transforms it into a normalized tensor.
    
    Args:
        path (Path): File path to the image.
        
    Returns:
        torch.Tensor: Tensor of shape [1, 100, 100] with values in range [0, 1].
    """
    try:
        # Convert to Grayscale ('L') and resize to fixed dimensions
        img = Image.open(path).convert("L").resize((100, 100), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)  # Returns tensor [1, 100, 100]
    except Exception as e:
        raise RuntimeError(f"[Read Error] {path} -> {e}")

def save_confusion_matrix_png(cm, class_names, carrier, model_name, run_tag, results_dir):
    """
    Renders and saves the confusion matrix as a PNG heatmap.
    """
    if cm is None: return None
    
    results_dir.mkdir(parents=True, exist_ok=True)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    
    # English labels for publication quality
    plt.title(f"Confusion Matrix - {carrier} ({model_name})")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    
    png_path = results_dir / f"cm_{carrier}_{model_name}_{run_tag}.png"
    plt.tight_layout()
    plt.savefig(png_path, dpi=180)
    plt.close()
    return png_path

def log_metrics_excel(excel_path, results_dir, class_names, metrics_dict):
    """
    Appends a row of experimental metrics to a summary Excel file.
    Handles file creation and appending to existing sheets.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Generate and save the confusion matrix image first
    cm_png = save_confusion_matrix_png(
        metrics_dict.get('cm'),
        class_names,
        metrics_dict['carrier'],
        metrics_dict['model_name'],
        metrics_dict['run_tag'],
        results_dir
    )
        
    row = {
        "timestamp": timestamp,
        "carrier": metrics_dict.get('carrier'),
        "model": metrics_dict.get('model_name'),
        "run_tag": metrics_dict.get('run_tag'),
        "num_classes": len(class_names),
        "val_acc": round(metrics_dict.get('acc', 0), 6),
        "va_loss": round(metrics_dict.get('loss', 0), 6),
        "f1_macro": round(metrics_dict.get('f1m', 0), 6),
        "f1_weighted": round(metrics_dict.get('f1w', 0), 6),
        "recall_macro": round(metrics_dict.get('recm', 0), 6),
        "epochs": metrics_dict.get('epochs'),
        "batch_size": metrics_dict.get('batch_size'),
        "lr": metrics_dict.get('lr'),
        "weight_decay": metrics_dict.get('weight_decay'),
        "notes": metrics_dict.get('notes', ""),
        "cm_png": str(cm_png) if cm_png else ""
    }
    
    df_row = pd.DataFrame([row])
    
    try:
        if excel_path.exists():
            # Use openpyxl to append without overwriting other sheets
            with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                # Load existing sheet to determine the next row
                prev_df = pd.read_excel(excel_path, sheet_name="resumen")
                df_row.to_excel(writer, sheet_name="resumen", index=False, header=False, startrow=len(prev_df) + 1)
        else:
            # Create new file
            df_row.to_excel(excel_path, sheet_name="resumen", index=False)
    except Exception as e:
        print(f"[WARNING] Could not append to existing Excel: {e}. Creating a new file.")
        df_row.to_excel(excel_path, sheet_name="resumen", index=False)
        
    print(f"Metrics successfully saved to: {excel_path}")

def save_all_checkpoints(model, arch, carrier_tag, out_dir, best_val_metric, monitor_metric, input_size=(1, 1, 100, 100), postfix="best"):
    """
    Exports model weights in multiple formats: .pth (State Dict), .pt (TorchScript), and .onnx.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = f"{arch}_{postfix}_{carrier_tag}"
    
    # 1. Save standard PyTorch state dictionary (.pth)
    pth_path = out / f"{base}.pth"
    torch.save(model.state_dict(), pth_path)
    
    # Prepare model for export (CPU mode)
    model_cpu = model.to("cpu").eval()
    example_input = torch.randn(*input_size)
    
    # 2. Save TorchScript format (.pt) - Useful for C++ production envs
    try:
        traced_script_module = torch.jit.trace(model_cpu, example_input)
        traced_script_module.save(str(out / f"{base}.pt"))
    except Exception as e:
        print(f"[WARNING] TorchScript export failed: {e}")
        
    # 3. Save ONNX format (.onnx) - Useful for interoperability
    try:
        torch.onnx.export(model_cpu, example_input, str(out / f"{base}.onnx"),
                          export_params=True, opset_version=12, do_constant_folding=True,
                          input_names=['input'], output_names=['logits'],
                          dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}})
    except ImportError:
        print("[INFO] 'onnx' library not installed. Skipping ONNX export.")
    except Exception as e:
        print(f"[WARNING] ONNX export failed: {e}")
        
    print(f"[SAVE] Checkpoint saved: {base} (val_{monitor_metric}={best_val_metric:.4f})")