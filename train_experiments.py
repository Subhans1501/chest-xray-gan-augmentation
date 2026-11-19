import os
import argparse
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image, make_grid
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.dataset import MedicalImageDataset
from src.models.dcgan import Generator, Discriminator
from src.metrics.evaluation import compute_fid, DownstreamClassifier


def train_gan_experiment(exp_name: str, mode: str, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nStarting {exp_name} (mode={mode}) on {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = MedicalImageDataset(
        root_dir=args.data_dir,
        img_size=args.img_size,
        channels=args.channels,
        mode=mode,
        sample_count=args.sample_count,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    netG = Generator(latent_dim=args.latent_dim, channels=args.channels, img_size=args.img_size).to(device)
    netD = Discriminator(channels=args.channels, img_size=args.img_size).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    fixed_noise = torch.randn(25, args.latent_dim, 1, 1, device=device)
    g_losses, d_losses = [], []

    for epoch in range(1, args.epochs + 1):
        netG.train()
        netD.train()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        for real_imgs, _ in dataloader:
            real_imgs = real_imgs.to(device)
            b_size = real_imgs.size(0)

            # One-sided label smoothing (0.9) helps keep discriminator gradients stable
            real_labels = torch.full((b_size, 1), 0.9, device=device)
            fake_labels = torch.full((b_size, 1), 0.0, device=device)

            netD.zero_grad()
            loss_d_real = criterion(netD(real_imgs), real_labels)

            noise = torch.randn(b_size, args.latent_dim, 1, 1, device=device)
            fake_imgs = netG(noise)
            loss_d_fake = criterion(netD(fake_imgs.detach()), fake_labels)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optimizerD.step()

            netG.zero_grad()
            loss_g = criterion(netD(fake_imgs), real_labels)
            loss_g.backward()
            optimizerG.step()

            epoch_d_loss += loss_d.item()
            epoch_g_loss += loss_g.item()

        avg_d = epoch_d_loss / len(dataloader)
        avg_g = epoch_g_loss / len(dataloader)
        d_losses.append(avg_d)
        g_losses.append(avg_g)

        if epoch % max(1, args.epochs // 4) == 0 or epoch == args.epochs:
            print(f"[{exp_name}] Epoch {epoch}/{args.epochs} | D: {avg_d:.4f} | G: {avg_g:.4f}")

    netG.eval()
    with torch.no_grad():
        fake_samples = (netG(fixed_noise).cpu() + 1.0) / 2.0
        save_image(make_grid(fake_samples, nrow=5, padding=2),
                   os.path.join(args.output_dir, f"{exp_name}_generated_grid.png"))

    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(netG.state_dict(), os.path.join(ckpt_dir, f"{exp_name}_generator.pth"))
    torch.save(netD.state_dict(), os.path.join(ckpt_dir, f"{exp_name}_discriminator.pth"))

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, args.epochs + 1), d_losses, label="Discriminator Loss", color="crimson", lw=1.8)
    plt.plot(range(1, args.epochs + 1), g_losses, label="Generator Loss", color="royalblue", lw=1.8)
    plt.title(f"{exp_name} - Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"{exp_name}_loss_curve.png"), dpi=200)
    plt.close()

    fid_score = compute_fid(
        dataloader, netG, device,
        num_samples=min(200, len(dataset)),
        latent_dim=args.latent_dim,
        in_channels=args.channels,
    )
    acc_real, acc_aug, f1_aug, auc_aug = evaluate_downstream_utility(dataset, netG, device, args)
    print(f"[{exp_name}] FID: {fid_score:.2f} | Real Acc: {acc_real*100:.1f}% | Aug Acc: {acc_aug*100:.1f}%")

    return {
        "experiment": exp_name,
        "mode": mode,
        "final_fid": round(fid_score, 2),
        "d_loss_final": round(d_losses[-1], 4),
        "g_loss_final": round(g_losses[-1], 4),
        "classifier_real_only_acc": round(acc_real * 100, 2),
        "classifier_augmented_acc": round(acc_aug * 100, 2),
        "classifier_augmented_f1": round(f1_aug, 4),
        "classifier_augmented_auc": round(auc_aug, 4),
        "accuracy_gain": round((acc_aug - acc_real) * 100, 2),
    }


def evaluate_downstream_utility(dataset, generator, device, args):
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
    crit = nn.CrossEntropyLoss()

    # Train baseline CNN on real scans only
    clf_real = DownstreamClassifier(in_channels=args.channels).to(device)
    opt_real = optim.Adam(clf_real.parameters(), lr=1e-3)
    clf_real.train()
    for _ in range(5):
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt_real.zero_grad()
            crit(clf_real(imgs), labels).backward()
            opt_real.step()

    clf_real.eval()
    preds, targets = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            preds.extend(clf_real(imgs.to(device)).argmax(dim=1).cpu().numpy())
            targets.extend(labels.numpy())
    acc_real = accuracy_score(targets, preds)

    # Train second CNN with GAN-generated samples mixed into training batches
    clf_aug = DownstreamClassifier(in_channels=args.channels).to(device)
    opt_aug = optim.Adam(clf_aug.parameters(), lr=1e-3)
    generator.eval()
    with torch.no_grad():
        synth_imgs = generator(torch.randn(120, args.latent_dim, 1, 1, device=device)).detach()
        synth_labels = torch.ones(120, dtype=torch.long, device=device)

    clf_aug.train()
    for _ in range(5):
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            half = imgs.size(0) // 2
            idx_sub = torch.randint(0, 120, (half,))
            mixed_imgs = torch.cat([imgs[:half], synth_imgs[idx_sub]], dim=0)
            mixed_labels = torch.cat([labels[:half], synth_labels[idx_sub]], dim=0)
            opt_aug.zero_grad()
            crit(clf_aug(mixed_imgs), mixed_labels).backward()
            opt_aug.step()

    clf_aug.eval()
    preds_aug, probs_aug, targets_aug = [], [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            out = clf_aug(imgs.to(device))
            probs_aug.extend(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
            preds_aug.extend(out.argmax(dim=1).cpu().numpy())
            targets_aug.extend(labels.numpy())

    acc_aug = accuracy_score(targets_aug, preds_aug)
    f1_aug = f1_score(targets_aug, preds_aug, zero_division=0)
    try:
        auc_aug = roc_auc_score(targets_aug, probs_aug)
    except ValueError:
        auc_aug = 0.0

    return acc_real, acc_aug, f1_aug, auc_aug


def main():
    parser = argparse.ArgumentParser(description="DCGAN Chest X-Ray Preprocessing Experiments")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--latent_dim", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--sample_count", type=int, default=400)
    parser.add_argument("--experiment", type=str, default="all", choices=["baseline", "clahe", "unsharp", "all"])
    args = parser.parse_args()

    experiments = []
    if args.experiment in ("baseline", "all"):
        experiments.append(("experiment_1_baseline", "baseline"))
    if args.experiment in ("clahe", "all"):
        experiments.append(("experiment_2_clahe", "clahe"))
    if args.experiment in ("unsharp", "all"):
        experiments.append(("experiment_3_unsharp", "unsharp"))

    all_results = {}
    for exp_name, mode in experiments:
        all_results[exp_name] = train_gan_experiment(exp_name, mode, args)

    with open(os.path.join(args.output_dir, "experiments_summary.json"), "w") as f:
        json.dump(all_results, f, indent=4)


if __name__ == "__main__":
    main()
