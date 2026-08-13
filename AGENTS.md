# AGENTS.md

## Project

This is an educational image-classification project using PyTorch.

The classifier distinguishes:
- bottle
- wristwatch
- pencil

## Technology

Use:
- Python 3
- PyTorch
- torchvision
- JupyterLab
- matplotlib

Do not use fastai.

## Development principles

This project is primarily for learning.

Prefer explicit PyTorch implementations over abstractions that hide the underlying mechanics.

Explain important ML concepts before implementing them.

Do not introduce unnecessary dependencies.

## Architecture

Reusable implementation belongs in `src/`.

Jupyter notebooks are for:
- exploration
- visualization
- experiments
- analysis

Do not put the entire application in a notebook.

## ML approach

Use transfer learning with a pretrained torchvision model.

Initially:
- freeze pretrained layers
- train the final classification layer

Later experiments may fine-tune deeper layers.

## Development workflow

Work incrementally.

Before moving to the next major step:
1. Implement the current step.
2. Run it.
3. Verify that it works.
4. Explain important concepts.
5. Show relevant results.

Do not generate the entire project at once.

## Important concepts

When relevant, explain:
- tensors
- Dataset
- DataLoader
- CNNs
- ResNet
- transfer learning
- logits
- softmax
- cross-entropy
- backpropagation
- optimizers
- learning rate
- epochs
- overfitting
- fine-tuning
- evaluation metrics