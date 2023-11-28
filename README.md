# CRN Synth
Synthetic data generation methods for the Cancer Registry of Norway

Contains example data and notebooks for test purposes.

## Installation
Create env:

>`conda create -n {env_name} python=3.9`

Install project dependencies:
>`pip install -r requirements.txt`


## Development
Code is automatically formatted and verified using pre-commit consisting of isort, black and flake8.

Make sure to run the following to set-up pre-commit hooks in your git repo.
>`pre-commit install`

When working on TSD use the [TSD-code-sync](https://marketplace.visualstudio.com/items?itemName=FlorianKrull.tsd-code-sync) plugin.