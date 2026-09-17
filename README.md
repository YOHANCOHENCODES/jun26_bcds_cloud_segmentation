## Segmentation of cloud regions

This repo is a Starting Pack for DS projects. You can rearrange the structure to make it fits your project.

### Development

For local Python development, with an active Python `venv`, first run this command:

```
pip install -r requirements-dev.txt
```

**NOTE:** The other file, `requirements.txt`, shouldn't be run on your dev machine. This one is needed for the GitHub/CLoud deployment of the Streamlit app.

### The Streamlit app

```
streamlit run src/streamlit/app.py
```

The app reads its data from `data/presentation/` — **always, also locally**, even when the full dataset sits next to it in `data/raw/`. `data/raw/` is ~1.3 GB and not part of the repository, so it is simply not there after a deployment; reading it locally would mean developing against data the deployed app never gets to see.

`data/presentation/` is committed and holds ~3 MB:

| | |
|---|---|
| `labels_slim.csv.gz` | all 22184 label rows without the RLE strings, plus `HasMask` and `MaskArea` |
| `labels_raw_head.csv` | the first rows of `train.csv` untouched |
| `train_labels.csv` + `train_images/` | a few labeled training images as examples |
| `test_images/` | unlabeled images for the model predictions |

Every number the app states about the dataset is computed from `labels_slim.csv.gz`, which covers the dataset completely — the images are examples and nothing is counted from them. The one fact that is in neither place, the number of test images, lives in `src/data/dataset_facts.py`.

Regenerate the directory from `data/raw/` after the raw data or the example selection changes:

```
python -m src.data.build_presentation_data
```

### Project Organization

```
    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Should be in your computer but not on Github (only in .gitignore)
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's name, and a short `-` delimited description, e.g.
    │                         `1.0-alban-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, links, and all other explanatory materials.
    │
    ├── reports            <- The reports that you'll make during this project as PDF
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   └── visualize.py
```

Project based on the [cookiecutter data science project template](https://drivendata.github.io/cookiecutter-data-science)

