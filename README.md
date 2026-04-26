# Deepfake Detection: Image and Video Pipeline

This repository contains a deep learning-based system for detecting deepfake content in both standalone images and video files. It leverages transfer learning with state-of-the-art Convolutional Neural Networks (CNNs) and automated face extraction to classify media as either "Real" or "Fake".

## 🌟 Key Features

* **Image Deepfake Detection**: Utilizes a fine-tuned `Xception` model with custom dense layers to classify static images.
* **Video Deepfake Detection**: Features an automated pipeline that extracts faces from video frames using `MTCNN` and classifies them using an `EfficientNetV2B0` architecture.
* **Mixed Precision & Multi-GPU Training**: The video model utilizes `mixed_float16` and `MirroredStrategy` for optimized, distributed training.
* **Interactive Inference**: Includes a ready-to-use Python script (`test.py`) for quick manual testing of images against the trained model.

## 📂 Repository Structure

* **`model.ipynb`**: Jupyter Notebook for training the image-based deepfake detector. It includes data loading, augmentation (rotation, shifting, flips), Xception model compilation, training, and evaluation.
* **`model.ipynb`**: Jupyter Notebook for training the video-based deepfake detector. It handles face extraction from video datasets (FaceForensics++ and Celeb-DF v2), sets up an EfficientNetV2B0 model, and performs a two-phase training process (frozen base followed by fine-tuning).
* **`test.py`**: A command-line inference script. It takes an image path as input, pre-processes it, and outputs the model's prediction along with a confidence score and a matplotlib visualization.
* **`dataset/`** *(Not included)*: Expected directory for training, validation, and test data.

## 📊 Datasets Used

The video pipeline is built to process standard deepfake datasets:
* **Celeb-DF v2** (Real and Synthesis sets)
* **FaceForensics++ C23** (Original, Deepfakes, Face2Face, FaceSwap, FaceShifter, NeuralTextures)

## 🛠️ Installation & Requirements

Ensure you have Python 3.x installed. Install the required dependencies using pip:

```bash
pip install numpy opencv-python tensorflow matplotlib seaborn scikit-learn mtcnn tqdm
```

## 🚀 Usage

* **1. Training the Image Model**
Open and run `model.ipynb`. Ensure your dataset is organized into `train`, `Sample_fake_images` (validation), and `test` directories inside a `./dataset` folder. The notebook will output a trained model named `image.keras`.

* **2. Training the Video Model**
Open and run `model.ipynb`. Update the `REAL_SOURCES` and `FAKE_SOURCES` paths to point to your local FaceForensics++ and Celeb-DF v2 datasets. The script will automatically extract faces using MTCNN, save them, and train the `EfficientNetV2B0` model, eventually outputting `video_model`.keras.

* **3. Testing an Image (Inference)**
Once image.keras is generated, you can test individual images using `test.py`.
```bash
python test.py
```
When prompted, enter the path to the image you want to test. The script will output the Raw Model Score, the final classification (Real/Fake), and display the analyzed image with the confidence percentage.

## 🧠 Model Architectures

**Image Classification** (`model.ipynb`)
* **Base Model:** Pre-trained Xception (ImageNet weights).

* **Custom Head:** Global Average Pooling -> Batch Normalization -> Dropout (0.5) -> Dense (512, ReLU) -> Batch Normalization -> Dropout (0.5) -> Dense (1, Sigmoid).

* **Loss Function:** Binary Crossentropy.

**Video Classification** (`model.ipynb`)
* **Face Extraction:** MTCNN (Multi-task Cascaded Convolutional Networks).

* **Base Model:** Pre-trained EfficientNetV2B0.

* **Custom Head:** Batch Normalization -> Dropout (0.5) -> Dense (128, ReLU) -> Dropout (0.3) -> Dense (2, Softmax).

* **Loss Function:** Sparse Categorical Crossentropy.
