# Pix2Pix-based Landscape Image Colorization  
A U-Net + Pix2Pix GAN model for generating RGB color images from grayscale landscape photos.

This project implements two models:
- **L1 baseline model** — deterministic, stable colorization.
- **Pix2Pix GAN model** — more vivid and realistic colorization with adversarial learning.

Both models are trained on landscape datasets and support inference via command line tools and a Web UI.

## Project Structure

project_root/
│
├── main.py # Train the Pix2Pix GAN model
├── no_GAN.py # Train the L1 baseline model
├── model_convolution.py # Inference model that supports arbitrary image resolutions
├── test_model.py # Batch colorization for 256×256 images
├── evaluate_model.py # Compute FID, LPIPS, SSIM, PSNR for model evaluation
├── evaluate_model_new.py # Compute NIQE, NIMA, Colorfulness, Class Consistency, LAB Histogram Overlap
│
├── split_dataset.py # Split dataset into train/val/test (70/20/10)
├── find_color.py # Retrieve missing color images for testset/gray
├── make_gray.py # Convert RGB images to grayscale in batch
│
├── webui.py # Simple Web UI for interactive colorization
│
├── models_select/ # Folder to store trained models (*.pth)
└── test_color_output/ # Default output folder for generated images



### Training Procedure ###

1. Install Python 3.10
2. Install dependencies

    Run the following command in your terminal:

    `pip install -r requirements.txt`
3. Place color images and grayscale images into `./dataset/color` and `./dataset/gray` respectively
4. Run `split_dataset.py`
5. Run `main.py` to train the model
6. Move the trained model into `./models_select/`
7. Run `test_model.py` or `model_convolution.py` for inference
8. View the generated results in `./test_color_output/`

