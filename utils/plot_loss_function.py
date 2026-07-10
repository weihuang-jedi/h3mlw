import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def plot_weather_ai_loss():
    # 1. Automatically locate the most recent PyTorch Lightning logger directory
    log_dirs = sorted(glob.glob("lightning_logs/version_*"))
    if not log_dirs:
        print("Error: No training log files found. Ensure train_single_gpu.py has started running.")
        return
        
    latest_log_dir = log_dirs[-1]
    csv_path = os.path.join(latest_log_dir, "metrics.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: Cannot find metrics.csv inside: {latest_log_dir}")
        return

    print(f"Reading training loss metrics history from: {csv_path}")
    df = pd.read_csv(csv_path)

    # 2. Extract and parse non-empty training metrics
    # FIX: Match the nested PyTorch Lightning naming scheme
    train_df = df[['epoch', 'train_loss_step_step']].dropna()
    val_df = df[['epoch', 'val_loss']].dropna()

    # Group multiple batch steps by epoch to create a clean, non-jagged line curve
    train_epoch = train_df.groupby('epoch')['train_loss_step_step'].mean()
    val_epoch = val_df.groupby('epoch')['val_loss'].mean()

    # 3. Initialize Plot Canvas
    plt.figure(figsize=(10, 6))
    
    # Plot curves
    plt.plot(train_epoch.index, train_epoch.values, label='Training Loss', marker='o', linewidth=2)
    plt.plot(val_epoch.index, val_epoch.values, label='Validation Loss', marker='s', linewidth=2)

    # Formatting metadata parameters
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Mean Squared Error (MSE Loss)', fontsize=12)
    plt.title('Global H3 Weather GATv2 Model - Training vs Validation Loss', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    
    # Lock X-axis steps strictly onto integers
    plt.xticks(train_epoch.index.astype(int))

    # 4. Save visualization chart to disk
    output_img = "h3_weather_loss_curves.png"
    plt.savefig(output_img, bbox_inches='tight', dpi=150)
    print(f"SUCCESS: Chart generated! Loss curves saved to: {output_img}")
    plt.show()

if __name__ == "__main__":
    plot_weather_ai_loss()

