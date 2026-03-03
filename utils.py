import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import os

plt.rcParams['font.sans-serif']  = ['SimHei']  
plt.rcParams['axes.unicode_minus']  = False



def create_exp_dir(path):
	"""Create experiment directory if it does not exist"""
	if not os.path.exists(path):
		os.makedirs(path)
	print('Experiment dir : {}'.format(path))



def set_random_seed(seed=42):
    """Set random seeds for reproducibility across multiple libraries."""    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disable benchmark for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.use_deterministic_algorithms(True)  # Force deterministic algorithms
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Avoid CUDA errors


def count_parameters_in_MB(model):
    """Calculate the number of model parameters in megabytes (MB)."""
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if v.requires_grad)/1e6


def get_trainable_parameters(model):
    params = filter(lambda p:p.requires_grad, model.parameters())
    return params



def plot_training_trend(train_list, val_list, title, save_dir):
    """Plot training and validation trends and save the figure"""
    # Create figure
    plt.figure(figsize=(8, 6), dpi=100)
    plt.plot(train_list, label='Training', c='#1f77b4')
    plt.plot(val_list, label='Validation', c='#ff7f0e')
    
    # Add plot elements
    plt.title(title, fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(title, fontsize=12)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Beautify the plot
    plt.gca().spines[['right', 'top']].set_visible(False)
    
    # Save and close
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{title}.png'))
    plt.close()
    

    
def plot_confusion_matrix(
    con_mat,
    save_path,
    class_names=None,
    normalize=False,
    title="Confusion Matrix",
    cmap="Blues",          
    vmax_quantile=0.99,    
    font_size=10
):
    """
    YOLO-like confusion matrix plot (bright colormap + grid + clear text).
    con_mat: (C, C) array-like
    class_names: list[str] or None
    normalize: normalize rows to sum=1
    """
    cm = np.array(con_mat, dtype=np.float32)

    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        cm = cm / row_sum

    C = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(C)]
    else:
        class_names = list(class_names)

    vmin = 0.0
    vmax = np.quantile(cm, vmax_quantile) if cm.size > 0 else 1.0
    if vmax <= 0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(1.2 * C + 3, 1.0 * C + 2))
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=font_size + 2)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=font_size)

    ax.set_xlabel("Predicted", fontsize=font_size + 1)
    ax.set_ylabel("True", fontsize=font_size + 1)

    ax.set_xticks(np.arange(C))
    ax.set_yticks(np.arange(C))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=font_size)
    ax.set_yticklabels(class_names, fontsize=font_size)

    ax.set_xticks(np.arange(-.5, C, 1), minor=True)
    ax.set_yticks(np.arange(-.5, C, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    fmt = ".2f" if normalize else "d"
    thresh = (vmax + vmin) * 0.5
    for i in range(C):
        for j in range(C):
            val = cm[i, j]
            text = format(val, fmt) if normalize else str(int(round(val)))
            ax.text(
                j, i, text,
                ha="center", va="center",
                fontsize=font_size,
                color="white" if val > thresh else "black"
            )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

