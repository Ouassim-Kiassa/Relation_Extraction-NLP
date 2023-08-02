

# Steps needed to run Mileston 1 code

- download csv file via data\dowload.sh script or with the following step and store it within "\data" folder:
```bash
wget wget https://www.dropbox.com/s/bkhxro9t81vjl6j/food_disease_dataset.csv -O food_disease_dataset.csv
```
- download python package seaborn
```bash
pip install seaborn
```

-run milestone1 jupyter notebook within "\docs" folder


# Install and Quick Start

First create a new conda environment with python 3.10 and activate it:

```bash
conda create -n tuwnlpie python=3.10
conda activate tuwnlpie
```

Then install this repository as a package, the `-e` flag installs the package in editable mode, so you can make changes to the code and they will be reflected in the package.

```bash
pip install -e .
```

All the requirements should be specified in the `setup.py` file with the needed versions. If you are not able to specify everything there
you can describe the additional steps here, e.g.:

Install `black` library for code formatting:
```bash
pip install black
```

Install `pytest` library for testing:
```bash
pip install pytest
```

## The directory structure and the architecture of the project

```
📦project-2022WS
 ┣ 📂data
 ┃ ┣ 📜README.md
 ┃ ┣ 📜bayes_model.tsv
 ┃ ┣ 📜bow_model.pt
 ┃ ┗ 📜imdb_dataset_sample.csv
 ┣ 📂docs
 ┃ ┗ 📜milestone1.ipynb
 ┣ 📂images
 ┃ ┗ 📜tuw_nlp.png
 ┣ 📂scripts
 ┃ ┣ 📜evaluate.py
 ┃ ┣ 📜predict.py
 ┃ ┗ 📜train.py
 ┣ 📂tests
 ┃ ┣ 📜test_milestone1.py
 ┃ ┣ 📜test_milestone2.py
 ┣ 📂tuwnlpie
 ┃ ┣ 📂milestone1
 ┃ ┃ ┣ 📜model.py
 ┃ ┃ ┗ 📜utils.py
 ┃ ┣ 📂milestone2
 ┃ ┃ ┣ 📜model.py
 ┃ ┃ ┗ 📜utils.py
 ┃ ┗ 📜__init__.py
 ┣ 📜.gitignore
 ┣ 📜LICENSE
 ┣ 📜README.md
 ┣ 📜setup.py
```

- `data`: This folder contains the data that you will use for training and testing your models. You can also store your trained models in this folder. The best practice is to store the data elsewhere (e.g. on a cloud storage) and provivde download links. If your data is small enough you can also store it in the repository.
- `docs`: This folder contains the reports of your project. You will be asked to write your reports here in Jupyter Notebooks or in simple Markdown files.
- `images`: This folder contains the images that you will use in your reports.
- `scripts`: This folder contains the scripts that you will use to train, evaluate and test your models. You can also use these scripts to evaluate your models.
- `tests`: This folder contains the unit tests for your code. You can use these tests to check if your code is working correctly.
- `tuwnlpie`: This folder contains the code of your project. This is a python package that is installed in the conda environment that you created. You can use this package to import your code in your scripts and in your notebooks. The `setup.py` file contains all the information about the installation of this repositorz. The structure of this folder is the following:
  - `milestone1`: This folder contains the code for the first milestone. You can use this folder to store your code for the first milestone.
  - `milestone2`: This folder contains the code for the second milestone. You can use this folder to store your code for the second milestone.
  - `__init__.py`: This file is used to initialize the `tuwnlpie` package. You can use this file to import your code in your scripts and in your notebooks.
- `setup.py`: This file contains all the information about the installation of this repository. You can use this file to install this repository as a package in your conda environment.
- `LICENSE`: This file contains the license of this repository.
- `team.cfg`: This file contains the information about your team.


```bash
black .
```
