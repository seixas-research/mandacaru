# Generating samples

## Generating samples with random noise

To create samples to train our model, we need first to generate structures
```python
from carcara.sampler.random_displacements import RandomDisplacements
from ase.build import bulk
from ase.calculator.emt import EMT

atoms = bulk("Au", "fcc", a=4.08, cubic=True).repeat([2,2,2])
calc = EMT()

generator = RandomDisplacements(atoms=atoms, calculator=calculator, seed=42)
generator.relax_structure(fmax=0.01, relax_cell=True)
generator.generate_samples(num_samples=100,
                           noise_type='uniform',
                           noise_level_pos=0.4,
                           noise_level_cell=0.4,
                           cell_mode='all')
generator.save_to_xyz("noisy_samples.xyz", compute_ref=True)

```

```{figure} ../_static/noisy.gif
---
width: 400px
name: Noisy samples.
align: center
---
Figure 1: Sample generation using uniform noise (0.4 Å) for atomic positions and lattice components.
```

## Calculating energies and forces with MACE


## Analyzing the diversity of the samples generated 


