
import os
import torch
from torchvision.utils import save_image
from model import Generator

Z_DIM = 128
OUTPUT_DIR = "outputs"
CHECKPOINT = os.path.join(OUTPUT_DIR, "stylegan_checkpoint.pth")

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G = Generator(Z_DIM).to(device)
checkpoint = torch.load(
    CHECKPOINT, map_location=device, weights_only=True
)
G.load_state_dict(checkpoint["generator"])
G.eval()

with torch.no_grad():
    # New latent vector for every generated face
    z = torch.randn(16, Z_DIM, device=device)
    images = G(z)

    # Save all faces in one grid
    save_image(
        images,
        os.path.join(OUTPUT_DIR, "generated_faces.png"),
        nrow=4,
        normalize=True,
        value_range=(-1, 1)
    )

    # Save every face separately
    for i in range(images.size(0)):
        save_image(
            images[i],
            os.path.join(OUTPUT_DIR, f"face_{i + 1}.png"),
            normalize=True,
            value_range=(-1, 1)
        )

print("Generated 16 images. Check the outputs folder.")
