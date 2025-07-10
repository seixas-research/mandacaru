<h1 align="center" style="margin-top:20px; margin-bottom:50px;">

<picture>
  <source srcset="https://raw.githubusercontent.com/leseixas/carcara/refs/heads/main/logo/logo_dark.png" media="(prefers-color-scheme: dark)">
  <source srcset="https://raw.githubusercontent.com/leseixas/carcara/refs/heads/main/logo/logo_light.png" media="(prefers-color-scheme: light)">
  <img src="https://raw.githubusercontent.com/leseixas/carcara/refs/heads/main/logo/logo_light.png" style="height: auto; width: auto; max-height: 100px; " alt="Carcará logo">
</picture>
</h1> 

[![License: MIT](https://img.shields.io/github/license/leseixas/carcara?color=green&style=for-the-badge)](LICENSE)    [![PyPI](https://img.shields.io/pypi/v/carcara?color=red&style=for-the-badge)](https://pypi.org/project/carcara/)

# Carcará

🚧 **(Under development)** 🚧

Towards Explainable, Scalable, and Accurate Machine-Learned Interatomic Potentials

# Installation

## From Pip
The easiest way to install Carcará is with pip:

```python
pip install carcara
```

# Getting started

## Training

```yaml

model: "MPNN"
name: "my_model"
training_dataset: "training.xyz"
validation_dataset: "validation.xyz"
test_dataset: "test.xyz"
cutoff_radius: 6.0
num_channels: 64
l_max: 1
mp_layers: 2
manybody_correlation: 3
energy_key: "REF_energy"
forces_key: "REF_forces"
stress_key: "REF_stress"
energy_weight: 10
forces_weight: 1000
stress_weight: 1000
seed: 42
device: cpu

```

## Evaluation

```python

# TODO

```

# License

This is an open source code under [MIT License](LICENSE).

# Acknowledgements

We thank financial support from FAPESP (Grant No. 2022/14549-3), INCT Materials Informatics (Grant No. 406447/2022-5), and CNPq (Grant No. 311324/2020-7).
