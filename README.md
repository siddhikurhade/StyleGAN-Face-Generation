# StyleGAN-Inspired Face Generation

A deep learning project that generates synthetic face images using a StyleGAN-inspired architecture built with PyTorch and trained on the CelebA dataset.

## Features

- Random latent-vector face generation
- Mapping network from latent space to style space
- Style mixing between two latent vectors
- Convolutional generator and discriminator
- Streamlit interface for generating and viewing images
- Save and download generated image grids

## Technologies

- Python
- PyTorch
- Torchvision
- Streamlit
- CelebA dataset

## Project Structure

```text
StyleGAN-Face-Generation/
├── model.py
├── train.py
├── generate.py
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
└── outputs/
```

## Setup

Install Python 3.10 or 3.11, then create and activate a virtual environment.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset

Download the CelebA dataset from the [official project website](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

Organize the extracted images so that `torchvision.datasets.ImageFolder` can load them, for example:

```text
data/celeba/
└── images/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

The training script expects images to be inside a class subfolder.

## Train

```bash
python train.py
```

## Generate Images

```bash
python generate.py
```

## Launch the Web App

```bash
python -m streamlit run app.py
```

## Limitations

This is an educational, StyleGAN-inspired implementation, not the original StyleGAN architecture. Image quality and diversity depend on training time, dataset size, hardware, and model stability.

## License

Add an appropriate license after checking the dataset and code licensing requirements.