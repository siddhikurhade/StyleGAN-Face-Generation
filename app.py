
import io
import streamlit as st
import torch
from PIL import Image
from torchvision.utils import make_grid
from model import Generator

st.set_page_config(
    page_title="StyleGAN Face Studio",
    page_icon="🎨",
    layout="centered"
)

st.title("StyleGAN Face Generation")
st.write("Generate synthetic faces and explore latent style mixing.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "outputs/stylegan_checkpoint.pth"
Z_DIM = 128


@st.cache_resource
def load_generator():
    generator = Generator(Z_DIM).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True
    )
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()
    return generator


try:
    G = load_generator()

    count = st.slider("Number of faces", 1, 16, 8)
    mode = st.selectbox(
        "Generation mode",
        ["Random faces", "Style mixing"]
    )

    if st.button("Generate Images", type="primary"):
        with torch.no_grad():
            z1 = torch.randn(count, Z_DIM, device=device)

            if mode == "Style mixing":
                z2 = torch.randn(count, Z_DIM, device=device)
                images = G(z1, mixing_z=z2, noise=True)
            else:
                images = G(z1, noise=True)

            images = images.cpu().clamp(-1, 1)
            images = (images + 1) / 2

            grid = make_grid(images, nrow=4, padding=2)
            grid = grid.permute(1, 2, 0).numpy()
            grid = (grid * 255).round().clip(0, 255).astype("uint8")

            output_image = Image.fromarray(grid).convert("RGB")
            buffer = io.BytesIO()
            output_image.save(buffer, format="PNG")

        st.image(output_image, caption=f"{mode} — {count} images")

        st.download_button(
            "Download generated image grid",
            data=buffer.getvalue(),
            file_name="generated_faces.png",
            mime="image/png"
        )

except FileNotFoundError:
    st.warning("Checkpoint not found. Run python train.py first.")

except Exception as e:
    st.error(f"Could not load or run the model: {e}")
