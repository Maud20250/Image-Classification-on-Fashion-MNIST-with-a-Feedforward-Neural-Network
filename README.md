# Assignment 12 — Neural Network and Deep Learning Basics

Image classification on **Fashion-MNIST** using a feedforward (fully-connected) neural network in TensorFlow/Keras.

**Author:** Jean Billa — Diploma in Data Analysis and Artificial Intelligence, Willis College

## Project Overview

This project builds, trains, and evaluates a simple feedforward neural network to classify images of clothing into 10 categories, and compares different architectures, optimizers, and data augmentation to understand what actually improves (or hurts) performance for this kind of model.

**Dataset:** [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) — 60,000 training images and 10,000 test images, 28x28 grayscale, 10 balanced classes:
`T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot`

## Repository Contents

| File | Description |
|---|---|
| `Assignment12_NeuralNetwork_FashionMNIST.ipynb` | Main Colab-ready notebook: data loading, preprocessing, model building, training, evaluation, and experiments |
| `report.pdf` | Two-page write-up summarizing methodology, results, and deployment discussion |
| `figures/` | All generated plots (sample images, training curves, confusion matrix, model comparison, augmentation comparison) |
| `results.json` | Raw numeric results (accuracy / precision / recall / F1 / classification reports) for every model trained |
| `train.py` | Standalone script version of the full pipeline (used to produce `results.json` and `figures/`) |

## How to Run

1. Open `Assignment12_NeuralNetwork_FashionMNIST.ipynb` in Google Colab (or Jupyter locally).
2. Run all cells top to bottom. The notebook will try to load Fashion-MNIST via `keras.datasets.fashion_mnist`, and automatically falls back to downloading the raw files from the official GitHub repo if that mirror is unreachable.
3. No GPU is required — the full notebook (baseline model + 3 architecture/optimizer experiments + augmentation experiment) runs in a few minutes on CPU.

### Dependencies
```
tensorflow
numpy
matplotlib
seaborn
scikit-learn
```

## Methodology Summary

- **Preprocessing:** pixel values scaled to `[0, 1]`; 6,000 images held out from the training set for validation; labels one-hot encoded.
- **Baseline architecture:** `Flatten → Dense(128, ReLU) → Dropout(0.2) → Dense(10, Softmax)`, compiled with Adam, categorical crossentropy, trained for 15 epochs (batch size 64).
- **Experiments:** a deeper network (256→128→64), a wider network (512 units), the same architecture trained with SGD instead of Adam, and a version with on-the-fly data augmentation (rotation, translation, zoom, horizontal flip).
- **Evaluation:** accuracy, macro precision/recall/F1, full classification report, and confusion matrix on the 10,000-image test set.

## Key Results

| Model | Test Accuracy | Macro F1 |
|---|---|---|
| Baseline (128 units, Adam) | 0.885 | 0.883 |
| Deeper (256-128-64) | 0.875 | 0.872 |
| **Wider (512)** | **0.886** | **0.884** |
| SGD optimizer (128) | 0.848 | 0.844 |
| Augmented baseline | 0.800 | 0.793 |

The baseline model trains in **well under a minute** on CPU. The widest network essentially matched the baseline as the best result; naive depth did not help a plain feedforward architecture; SGD underperformed Adam within the same epoch budget; and data augmentation *hurt* performance here because a Dense-only network has no built-in spatial invariance to exploit — that combination is expected to pay off with a convolutional architecture instead.

Most misclassifications occur between visually similar upper-body garments: **Shirt, T-shirt/top, Pullover, and Coat** — exactly where the lack of spatial feature extraction in a fully-connected network is most limiting.

## Practical Application

See Section 8 of the notebook / the report for a discussion of a hypothetical deployment scenario: automated product-image tagging for an online fashion retailer's catalog, including scalability, real-time processing, and integration considerations.

## Future Work

- Replace Dense layers with a small CNN (Conv2D + MaxPooling) to better exploit spatial structure and close the gap on the confused garment classes.
- Add learning-rate scheduling and early stopping.
- Re-test data augmentation paired with a CNN and a longer training schedule.
- Validate on real (non-benchmark) product photos before considering production use.
