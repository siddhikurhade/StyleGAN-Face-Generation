
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from model import Generator, Discriminator

# Configuration
DATA_DIR = "data/celeba"
OUTPUT_DIR = "outputs"
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "stylegan_checkpoint.pth")

IMAGE_SIZE = 64
BATCH_SIZE = 4          # Lower memory usage for CPU
Z_DIM = 128
EPOCHS = 20             # Number of additional epochs per run
LR = 0.0002
RESUME_TRAINING = True  # Continue from an existing checkpoint

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Reproducibility
torch.manual_seed(42)

# Dataset preprocessing
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

if len(dataset) < BATCH_SIZE:
    raise ValueError(
        f"Dataset has only {len(dataset)} images; "
        f"at least {BATCH_SIZE} are required."
    )

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    drop_last=True
)

print("Dataset images:", len(dataset))
print("Batches per epoch:", len(loader))

# Initialize models
G = Generator(Z_DIM).to(device)
D = Discriminator().to(device)

criterion = nn.BCEWithLogitsLoss()

opt_G = optim.Adam(
    G.parameters(), lr=LR, betas=(0.5, 0.999)
)
opt_D = optim.Adam(
    D.parameters(), lr=LR, betas=(0.5, 0.999)
)

start_epoch = 0

# Resume model and optimizer states when available
if RESUME_TRAINING and os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True
    )

    G.load_state_dict(checkpoint["generator"])
    D.load_state_dict(checkpoint["discriminator"])

    if "optimizer_G" in checkpoint:
        opt_G.load_state_dict(checkpoint["optimizer_G"])

    if "optimizer_D" in checkpoint:
        opt_D.load_state_dict(checkpoint["optimizer_D"])

    start_epoch = checkpoint.get("epoch", 0)

    print(f"Resuming from completed epoch {start_epoch}")

else:
    print("Starting training from scratch.")

# Training loop
for epoch in range(start_epoch, start_epoch + EPOCHS):
    G.train()
    D.train()

    total_g_loss = 0.0
    total_d_loss = 0.0

    for step, (real, _) in enumerate(loader):
        real = real.to(device)
        batch = real.size(0)

        real_labels = torch.ones(batch, 1, device=device)
        fake_labels = torch.zeros(batch, 1, device=device)

        # -------------------------
        # Train Discriminator
        # -------------------------
        z = torch.randn(batch, Z_DIM, device=device)

        with torch.no_grad():
            fake = G(z)

        opt_D.zero_grad(set_to_none=True)

        real_output = D(real)
        fake_output = D(fake)

        real_loss = criterion(real_output, real_labels)
        fake_loss = criterion(fake_output, fake_labels)
        d_loss = real_loss + fake_loss

        d_loss.backward()
        opt_D.step()

        # -------------------------
        # Train Generator
        # -------------------------
        z = torch.randn(batch, Z_DIM, device=device)

        opt_G.zero_grad(set_to_none=True)

        generated = G(z)
        generated_output = D(generated)
        g_loss = criterion(generated_output, real_labels)

        g_loss.backward()
        opt_G.step()

        total_g_loss += g_loss.item()
        total_d_loss += d_loss.item()

        if step % 100 == 0:
            print(
                f"Epoch [{epoch + 1}/{start_epoch + EPOCHS}] "
                f"Step [{step + 1}/{len(loader)}] "
                f"D Loss: {d_loss.item():.4f} "
                f"G Loss: {g_loss.item():.4f}"
            )

    avg_g_loss = total_g_loss / len(loader)
    avg_d_loss = total_d_loss / len(loader)

    print(
        f"Epoch {epoch + 1} completed | "
        f"Average D Loss: {avg_d_loss:.4f} | "
        f"Average G Loss: {avg_g_loss:.4f}"
    )

    # Save sample faces
    G.eval()
    with torch.no_grad():
        z = torch.randn(16, Z_DIM, device=device)
        samples = G(z, noise=False)

        sample_path = os.path.join(
            OUTPUT_DIR, f"epoch_{epoch + 1:03d}.png"
        )

        save_image(
            samples.cpu(),
            sample_path,
            nrow=4,
            normalize=True,
            value_range=(-1, 1)
        )

    # Save models and optimizer states
    torch.save({
        "generator": G.state_dict(),
        "discriminator": D.state_dict(),
        "optimizer_G": opt_G.state_dict(),
        "optimizer_D": opt_D.state_dict(),
        "epoch": epoch + 1,
        "g_loss": avg_g_loss,
        "d_loss": avg_d_loss,
    }, CHECKPOINT_PATH)

    print("Saved samples:", sample_path)
    print("Checkpoint saved:", CHECKPOINT_PATH)

print("Training completed.")
