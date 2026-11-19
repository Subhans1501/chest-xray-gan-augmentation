import os
import argparse
import torch
from torchvision.utils import save_image, make_grid
from src.models.dcgan import Generator


def generate_synthetic_scans(model_path: str, num_images: int = 16,
                             output_file: str = "outputs/generated_samples.png",
                             latent_dim: int = 100, img_size: int = 64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: '{model_path}'.")

    netG = Generator(latent_dim=latent_dim, channels=1, img_size=img_size).to(device)
    netG.load_state_dict(torch.load(model_path, map_location=device))
    netG.eval()

    with torch.no_grad():
        z = torch.randn(num_images, latent_dim, 1, 1, device=device)
        fake_imgs = (netG(z).cpu() + 1.0) / 2.0
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        save_image(make_grid(fake_imgs, nrow=4, padding=2), output_file)
        print(f"Saved {num_images} generated scans to '{output_file}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic chest X-rays from a saved DCGAN checkpoint")
    parser.add_argument("--model_path", type=str, default="outputs/checkpoints/experiment_1_baseline_generator.pth")
    parser.add_argument("--num_images", type=int, default=16)
    parser.add_argument("--output_file", type=str, default="outputs/generated_samples.png")
    args = parser.parse_args()
    generate_synthetic_scans(args.model_path, args.num_images, args.output_file)
