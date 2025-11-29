"""
Final Model Evaluation Script
-----------------------------
Executes the definitive performance assessment of the trained Student models (ResNet18/ResNet50)
against the isolated Test Set.

Key artifacts generated:
- Confusion Matrix plots.
- Detailed classification reports (Precision, Recall, F1 per class).
- Master Excel log entry with final test metrics.
"""

import sys
import torch
import numpy as np
import datetime
import traceback
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score

# --- Local Modules ---
import config
import models
import utils

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Models to evaluate (must match those trained in step 03)
MODELS_TO_EVALUATE = ['resnet18', 'resnet50'] 

# ==============================================================================
# Helper Classes & Functions
# ==============================================================================

class LabeledSpectro(Dataset):
    """Dataset wrapper for loading spectrogram images and labels."""
    def __init__(self, files, labels, transform=None):
        self.files = files
        self.labels = labels
        self.transform = transform

    def __len__(self): 
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        try:
            x = utils.load_png_gray(path)
            if self.transform: 
                x = self.transform(x)
            y = self.labels[i]
            return x, y
        except Exception as e:
            print(f"[WARN] Skipping file due to error: {e}")
            # Return dummy tensor to preserve batch structure
            return torch.zeros(1, config.IMG_SIZE[0], config.IMG_SIZE[1], dtype=torch.float32), self.labels[i]

def maybe_resize_for_resnet(x, should_resize):
    """Resizes input tensor to 224x224 if required by the architecture."""
    if should_resize:
        return torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    return x

def evaluate_on_test(model, loader, params):
    """Runs inference on the test loader and returns ground truth and predictions."""
    device = torch.device(config.DEVICE)
    model.eval()
    preds, gts = [], []
    
    with torch.no_grad():
        for xb, yb in loader:
            xb = maybe_resize_for_resnet(xb, params.get('RESIZE_224', False))
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds.append(logits.softmax(1).argmax(1).cpu())
            gts.append(yb.cpu())
    
    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(gts).numpy()
    return y_true, y_pred

# ==============================================================================
# Main Execution Flow
# ==============================================================================
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"STARTING FINAL COMPARATIVE EVALUATION (TEST SET)")
    print(f"Target Scenario: {config.CURRENT_CARRIER}")
    print(f"{'='*60}\n")
    
    # 1. Load Test Set (Common for all models)
    if not config.TEST_DIR.exists():
        print(f"[CRITICAL] Test directory not found at:")
        print(f"   {config.TEST_DIR}")
        print("   Please execute 00_prepare_data.py first.")
        sys.exit(1)

    test_files, test_labels = [], []
    class_names = sorted([p.name for p in config.TEST_DIR.iterdir() if p.is_dir()])
    cls2idx = {name: i for i, name in enumerate(class_names)}
    num_classes = len(class_names)

    for class_name in class_names:
        class_path = config.TEST_DIR / class_name
        files = list(class_path.glob("*.png"))
        test_files.extend(files)
        test_labels.extend([cls2idx[class_name]] * len(files))

    if len(test_files) == 0:
        print("[CRITICAL] No images found in TEST directory.")
        sys.exit(1)

    test_ds = LabeledSpectro(test_files, test_labels, transform=None)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=config.NUM_WORKERS)
    
    print(f"[DATA] Test Dataset Loaded: {len(test_ds)} samples.")
    print(f"[DATA] Classes: {class_names}")
    print("-" * 60)

    # 2. Evaluation Loop
    for model_key in MODELS_TO_EVALUATE:
        print(f"\n[INFO] EVALUATING MODEL: Student {model_key.upper()}...")
        
        try:
            params = config.TRAIN_PARAMS[model_key]
            
            # Construct folder name based on ThesisHelper convention:
            # "student_trained_with_win_teachers" + model_name
            folder_name = f"student_trained_with_win_teachers{model_key}"
            
            # Use relative path from config
            ckpt_path = config.ARTIFACTS_DIR / folder_name / 'best_model.pth'
            
            print(f"   Expected Checkpoint: {ckpt_path}")
            
            if not ckpt_path.exists():
                print(f"[WARN] 'best_model.pth' not found.")
                print(f"       Verify that {model_key} was trained successfully.")
                continue

            # Load Architecture
            device = torch.device(config.DEVICE)
            model = models.make_model(
                params['MODEL_NAME'], 
                num_classes,
                params.get('USE_PRETRAIN', True)
            ).to(device)
            
            # Load Weights
            # Handle potential weight loading issues across versions
            try:
                model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            except TypeError:
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
                
            print(f"   Model loaded successfully.")

            # 3. Execute Evaluation
            y_true, y_pred = evaluate_on_test(model, test_loader, params)

            # 4. Compute Metrics
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
            
            print(f"   [RESULT] Test Accuracy: {acc:.4f} | Test F1-Macro: {f1:.4f}")
            
            # 5. Generate Artifacts
            cm = confusion_matrix(y_true, y_pred)
            report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
            
            # Save Confusion Matrix Image
            utils.save_confusion_matrix_png(
                cm, 
                class_names, 
                config.CURRENT_CARRIER, 
                f"FINAL_TEST_student_{model_key}", 
                "final_evaluation", 
                config.ARTIFACTS_DIR
            )

            # Save Text Report
            report_path = config.ARTIFACTS_DIR / f"FINAL_TEST_REPORT_student_{model_key}.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"--- FINAL TEST REPORT (MODEL: {model_key}) ---\n")
                f.write(f"Scenario: {config.CURRENT_CARRIER}\n")
                f.write(f"Date: {datetime.datetime.now()}\n")
                f.write(f"Source Checkpoint: {ckpt_path}\n\n")
                f.write(report)
                f.write("\n\nConfusion Matrix:\n")
                f.write(np.array2string(cm))
            
            # 6. Log to Excel
            final_metrics = {
                'carrier': config.CURRENT_CARRIER,
                'model_name': f"student_{model_key}", 
                'run_tag': "FINAL_TEST_SET", # Distinct tag for filtering in Excel
                'acc': acc,
                'f1m': f1,
                'f1w': f1_score(y_true, y_pred, average='weighted', zero_division=0),
                'recm': rec,
                'cm': cm,
                'lr': params['LR'],
                'weight_decay': params.get('WEIGHT_DECAY', 0.0),
                'notes': 'Official Evaluation on Isolated Test Set'
            }
            utils.log_metrics_excel(config.METRICS_FILE, config.ARTIFACTS_DIR, class_names, final_metrics)
            print(f"   Metrics logged to Master Excel.")
            
        except Exception as e:
            print(f"[ERROR] Failed to evaluate {model_key}: {e}")
            traceback.print_exc()

    print("\n" + "="*60)
    print("CARRIER EVALUATION COMPLETED")
    print("Please review the consolidated Excel file for R18 vs R50 comparison.")
    print("="*60)