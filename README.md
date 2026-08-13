# Image Classifier

Educational PyTorch project for classifying images of bottles, wristwatches,
and pencils.

## Project layout

- `data/train/` — training images, organized into one directory per class.
- `data/validation/` — validation images, using the same class directories.
- `notebooks/exploration.ipynb` — interactive data exploration and visual analysis.
- `src/dataset.py` — reusable dataset and data-loading code (later step).
- `src/transforms.py` — separate training and validation image transformations.
- `src/model.py` — pretrained ResNet18 construction and classifier replacement.
- `src/forward_pass.py` — one-batch forward-pass demonstration.
- `src/train.py` — explicit training and validation loops.
- `src/evaluate.py` — evaluation and metrics code (later step).
- `requirements.txt` — Python dependencies for the project environment.
- `PLAN.md` — checklist of the incremental learning steps.
- `TASKS.md` — detailed project requirements.

The source modules and notebook are intentionally placeholders at this stage.
Model and training implementation will be added in later steps.

## Dataset layout

Place images beneath the matching class directory in both dataset splits:

```text
data/
├── train/
│   ├── bottle/
│   ├── pencil/
│   └── wristwatch/
└── validation/
    ├── bottle/
    ├── pencil/
    └── wristwatch/
```

The project uses `torchvision.datasets.ImageFolder`. It discovers class names
from the immediate subdirectory names, sorts them alphabetically, and assigns
the sorted positions as numeric labels. Therefore the reproducible mapping is
`bottle: 0`, `pencil: 1`, and `wristwatch: 2`. The loader prints this mapping
explicitly when run with:

```bash
python -m src.dataset
```

The training and validation directories must contain the same three class
folders. `src/dataset.py` connects these datasets to the transformation
pipelines and creates the `DataLoader`s used in later steps.

## Image transformations

`src/transforms.py` defines two pipelines. Training images use a moderate
random resized crop and horizontal flip, which create plausible variations of
the object without aggressively changing its identity. Validation images use a
deterministic resize and center crop, so repeated evaluation sees the same
input. Both pipelines convert images to tensors and normalize RGB channels
with the mean and standard deviation used for ImageNet-pretrained models.

## Dataset and DataLoader batching

`src/dataset.py` creates a training `Dataset`, a validation `Dataset`, and one
`DataLoader` for each. A `Dataset` represents individual image/label samples;
a `DataLoader` groups them into batches. The training loader shuffles samples,
while the validation loader does not, so validation is repeatable. `batch_size`
and `num_workers` are configurable, with defaults of 32 and 0.

For example, a batch shape of `[32, 3, 224, 224]` means 32 images, 3 RGB
channels, and 224 pixels in both height and width. Inspect a batch with:

```bash
python -m src.dataset --batch-size 32 --num-workers 0
```

## Transfer-learning model

`src/model.py` creates a pretrained ResNet18. ResNet is a convolutional neural
network architecture with residual connections that help information and
gradients pass through many layers. The pretrained weights were learned on
ImageNet, a large labeled image dataset, so the network already contains
general visual features such as edges, textures, and shapes.

This is transfer learning: we reuse those learned features for the bottle,
pencil, and wristwatch task. The original final layer is replaced with a new
linear layer whose output has exactly three values, one logit per class. The
pretrained layers are frozen initially, so only this new classification layer
is trained. The output values are logits, not probabilities; the later loss
and inference steps will explain how to use them.

## One forward pass

`src/forward_pass.py` takes one batch from the training `DataLoader` and passes
it through the model. The output is a matrix of logits: each row belongs to one
image and its three values are the model's unnormalized scores for the three
classes. `logits.argmax(dim=1)` selects the index of the largest score for each
image, which gives the predicted class index. This step uses `model.eval()` and
`torch.no_grad()` because it is an inspection pass, not parameter training.

## Training loop

`src/train.py` implements the learning steps explicitly. For each training
batch it moves images and labels to the device, clears old gradients, runs the
forward pass, calculates `CrossEntropyLoss`, backpropagates with
`loss.backward()`, and updates the trainable parameters with Adam.

The reported loss measures how far the logits are from the correct labels;
lower is generally better. Accuracy is the fraction of correctly classified
images. Training metrics describe the data used for updates, while validation
metrics measure generalization on held-out data. The training loop accepts an
explicit device; automatic hardware selection is handled in the next step.

## Environment

Create and activate the project virtual environment, then install the declared
dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The environment is kept out of version control. The project uses the standard
Python packages listed in `requirements.txt`; PyTorch selects the available
hardware at runtime in later steps.
