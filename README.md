# Image Classifier

Educational PyTorch project for classifying images of bottles, wristwatches,
and pencils.

## Project layout

- `data/train/` — training images, organized into one directory per class.
- `data/validation/` — validation images, using the same class directories.
- `notebooks/exploration.ipynb` — interactive data exploration and visual analysis.
- `src/dataset.py` — reusable dataset and data-loading code (later step).
- `src/model.py` — model construction code (later step).
- `src/train.py` — training-loop code (later step).
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
folders. Image transformations and `DataLoader` batching will be added in
later steps.

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
