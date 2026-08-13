Build a small image-classification project in Python using PyTorch.

The goal is to classify images into exactly three classes:

- bottle
- wristwatch
- pencil

I want this project primarily as a learning exercise, so favor clear, explicit implementations over abstractions that hide what is happening.

Use:
- Python 3
- PyTorch
- torchvision
- JupyterLab for experimentation
- regular Python modules for reusable/project code
- matplotlib for visualization

Do NOT use fastai for this project.

The project should use transfer learning with a pretrained torchvision model rather than training a CNN from scratch.

Implement the project incrementally in the following order.

STEP 1 - Create the project

Create this structure:

image-classifier/
├── data/
│   ├── train/
│   │   ├── bottle/
│   │   ├── wristwatch/
│   │   └── pencil/
│   └── validation/
│       ├── bottle/
│       ├── wristwatch/
│       └── pencil/
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── requirements.txt
└── README.md

Explain what each directory/file is responsible for.

STEP 2 - Set up the Python environment

Create a virtual environment and install the required dependencies.

Use the appropriate PyTorch installation for the user's machine where possible.

The initial dependencies should be approximately:

torch
torchvision
jupyterlab
matplotlib
numpy
pillow

Do not add unnecessary dependencies.

STEP 3 - Prepare the dataset

Use torchvision.datasets.ImageFolder.

Explain how ImageFolder determines class labels from directory names.

Load:

data/train/
data/validation/

Print:

- number of training images
- number of validation images
- class names
- class-to-index mapping

Make sure the class mapping is explicitly displayed.

STEP 4 - Explore the data in JupyterLab

Create notebooks/exploration.ipynb.

Use the notebook to:

1. Load the dataset.
2. Display several example images.
3. Display examples from each class.
4. Print image dimensions.
5. Calculate the number of images in each class.
6. Plot the class distribution.
7. Explain why image dimensions need to be normalized before being passed to a neural network.

Keep this notebook focused on exploration and understanding.

STEP 5 - Create image transformations

Use torchvision.transforms.

Create separate transformations for:

training:
- resize/crop as appropriate
- data augmentation
- conversion to tensor
- normalization

validation:
- deterministic resize/crop
- conversion to tensor
- normalization

Explain why training and validation transformations should differ.

Do not introduce aggressive augmentation without explaining why it is appropriate.

STEP 6 - Create the datasets and DataLoaders

Implement this in src/dataset.py.

Create:

- training Dataset
- validation Dataset
- training DataLoader
- validation DataLoader

Use a configurable batch size.

Explain:

- Dataset
- DataLoader
- batch
- shuffle
- num_workers

Print a batch and inspect its tensor shape.

For example, explain what something like:

[32, 3, 224, 224]

means.

STEP 7 - Create the model

Implement this in src/model.py.

Use a pretrained torchvision model such as ResNet18.

Do NOT train the entire network initially.

Replace the final classification layer so that it outputs exactly 3 classes.

Explain:

- what ResNet is
- what pretrained means
- what ImageNet means
- what transfer learning means
- what the final classification layer does
- why the output has three values

Initially freeze the pretrained layers and train only the final classification layer.

STEP 8 - Understand one forward pass

Before implementing the complete training loop, demonstrate a single forward pass.

Take one batch:

images, labels = next(iter(train_loader))

Pass the images through the model.

Print:

- input shape
- output shape
- labels
- predicted class indices

Explain that the model's output consists of logits rather than probabilities.

Explain how the predicted class is obtained from the logits.

STEP 9 - Implement the training loop

Implement the training loop in src/train.py.

Use:

CrossEntropyLoss

and an appropriate optimizer such as Adam.

The training loop should explicitly perform:

1. Load a batch.
2. Move tensors to the device.
3. Clear gradients.
4. Perform forward pass.
5. Calculate loss.
6. Perform backward pass.
7. Update model parameters.

Do not hide these steps behind a training framework.

Track for every epoch:

- training loss
- training accuracy
- validation loss
- validation accuracy

Print these metrics after each epoch.

Explain what each metric means.

STEP 10 - Use the appropriate hardware

Detect whether the machine has:

- CUDA
- Apple Silicon MPS
- CPU

Select the best available device.

Print the selected device.

Do not assume CUDA because the project may be running on a Mac.

STEP 11 - Save the trained model

After training, save the model's state_dict.

For example:

models/best_model.pth

Save the model when validation accuracy improves.

Also save the class-to-index mapping or otherwise make sure the mapping is reproducible.

Explain why state_dict is preferable to blindly serializing the entire model object.

STEP 12 - Evaluate the model

Implement evaluation in src/evaluate.py.

Calculate:

- validation accuracy
- precision
- recall
- F1 score
- confusion matrix

Display the confusion matrix.

Explain which classes the model confuses most frequently.

Do not rely exclusively on accuracy.

STEP 13 - Visualize predictions

Create a notebook section that displays validation images with:

- actual class
- predicted class
- confidence

Show both correct and incorrect predictions.

Explain that this is useful for understanding model failure modes.

STEP 14 - Test inference on a single image

Implement a function that accepts:

image_path

and returns:

- predicted class
- class probabilities
- confidence

Use the trained model in evaluation mode.

Explain why:

model.eval()

is necessary during inference.

Also explain the role of:

torch.no_grad()

STEP 15 - Experiment with fine-tuning

After the initial model works, modify the project to support fine-tuning.

First train only the final layer.

Then unfreeze some of the later ResNet layers and train them with a smaller learning rate.

Compare:

1. Training only the classifier.
2. Fine-tuning part of the pretrained network.

Explain why fine-tuning can improve performance.

STEP 16 - Experiment with the dataset

Add experiments for:

- different batch sizes
- different learning rates
- different augmentation strategies
- different numbers of epochs

Record the results.

Do not perform experiments blindly. Explain what hypothesis each experiment is testing.

STEP 17 - Add a simple prediction script

Create:

src/predict.py

It should allow:

python src/predict.py path/to/image.jpg

and print something similar to:

Prediction: bottle
Confidence: 0.94

Also print the probability for each class.

STEP 18 - Clean up the project

Move reusable logic out of the notebook and into src/.

The notebook should primarily contain:

- exploration
- visualization
- experiments
- analysis

The Python modules should contain reusable implementation.

Avoid duplicating code between the notebook and Python modules.

STEP 19 - README

Create a README explaining:

1. What the project does.
2. How to install dependencies.
3. How to structure the dataset.
4. How to train the model.
5. How to evaluate it.
6. How to run inference.
7. What transfer learning means.
8. What model was used.
9. What metrics were obtained.

STEP 20 - Teaching requirement

This is an educational project.

At every major step, explain the underlying machine-learning concept before implementing it.

In particular, I want to understand:

- tensors
- Dataset
- DataLoader
- batches
- image normalization
- augmentation
- CNNs
- ResNet
- pretrained models
- transfer learning
- logits
- softmax
- cross-entropy loss
- gradients
- backpropagation
- optimizer
- learning rate
- epochs
- training vs validation
- overfitting
- fine-tuning
- accuracy
- precision
- recall
- F1
- confusion matrix

Do not just produce code that works. Make the implementation understandable.

IMPORTANT DEVELOPMENT STYLE:

Work incrementally.

After completing each major step:
1. Show me what was created/changed.
2. Explain the important concepts.
3. Run or provide a small test proving that the step works.
4. Only then proceed to the next step.

Do not generate the entire project as one giant block of code.

Avoid unnecessary frameworks and abstractions.

Prefer PyTorch and torchvision primitives so I can understand what is actually happening.