import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def create_unified_visualizations(output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    experiments = [
        ("experiment_1_baseline", "Exp 1: Baseline DCGAN (Raw)"),
        ("experiment_2_clahe", "Exp 2: CLAHE Preprocessing"),
        ("experiment_3_unsharp", "Exp 3: Unsharp Masking"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (exp_key, title) in zip(axes, experiments):
        img_path = os.path.join(output_dir, f"{exp_key}_loss_curve.png")
        if os.path.exists(img_path):
            ax.imshow(mpimg.imread(img_path))
            ax.axis("off")
            ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparative_loss_curves.png"), dpi=200)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (exp_key, title) in zip(axes, experiments):
        img_path = os.path.join(output_dir, f"{exp_key}_generated_grid.png")
        if os.path.exists(img_path):
            ax.imshow(mpimg.imread(img_path))
            ax.axis("off")
            ax.set_title(f"Synthetic Outputs - {title}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparative_generated_samples.png"), dpi=200)
    plt.close()


if __name__ == "__main__":
    create_unified_visualizations()
