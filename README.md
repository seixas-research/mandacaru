<h1 align="center" style="margin-top:20px; margin-bottom:50px;">

<a href="https://github.com/seixasgroup/carcara" target="_blank" rel="noopener noreferrer">
  <picture>
    <source srcset="https://raw.githubusercontent.com/seixasgroup/carcara/refs/heads/main/logo/logo_dark.png" media="(prefers-color-scheme: dark)">
    <source srcset="https://raw.githubusercontent.com/seixasgroup/carcara/refs/heads/main/logo/logo_light.png" media="(prefers-color-scheme: light)">
    <img src="https://raw.githubusercontent.com/seixasgroup/carcara/refs/heads/main/logo/logo_light.png" alt="Carcará logo" style="height: auto; width: auto; max-height: 100px;">
  </picture>
</a>
</h1> 

[![License: MIT](https://img.shields.io/github/license/seixasgroup/carcara?color=green&style=for-the-badge)](LICENSE)    [![PyPI](https://img.shields.io/pypi/v/carcara?color=red&style=for-the-badge)](https://pypi.org/project/carcara/)

# Carcará

🚧 **(Under development)** 🚧

Towards Explainable, Scalable, and Accurate Machine-Learned Interatomic Potentials

# Installation

## From pip
The easiest way to install Carcará is with pip:

```python
pip install carcara
```

## From github
To install Carcará directly from the GitHub repository, run the following commands:

```python
pip install git+https://github.com/seixasgroup/carcara.git
```



# Getting started

## Training

```yaml

model: "MACE"
name: "my_model"

datasets:
  training: "training.xyz"
  validation: "validation.xyz"
  test: "test.xyz"

e3nn_irreps:
  num_channels: 64
  l_max: 1

cutoff_radius: 6.0
message_passing_layers: 2
manybody_correlation: 3

training_attributes:
  energy: "REF_energy"
  forces: "REF_forces"

weights:
  energy: 10
  forces: 1000

seed: 42
device: cpu

```

## Evaluation

```python

# TODO

```

# License

This is an open source code under [MIT License](https://raw.githubusercontent.com/seixasgroup/carcara/refs/heads/main/LICENSE).

# Acknowledgements

We thank financial support from FAPESP (Grant No. 2022/14549-3), INCT Materials Informatics (Grant No. 406447/2022-5), and CNPq (Grant No. 311324/2020-7).
