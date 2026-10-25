
import torch
from model import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G = Generator(128).to(device)
checkpoint = torch.load(
    "outputs/stylegan_checkpoint.pth",
    map_location=device,
    weights_only=True
)
G.load_state_dict(checkpoint["generator"])
G.eval()

with torch.no_grad():
    z = torch.randn(4, 128, device=device)
    images = G(z, noise=False)

    for i in range(1, 4):
        difference = (images[0] - images[i]).abs().mean().item()
        print(f"Face 1 vs Face {i + 1}: {difference:.8f}")
