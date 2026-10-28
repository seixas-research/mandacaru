# Tutorials

Follow the first three tutorials for native VQE and ADAPT on LiH, followed by
a short molecular DFT geometry scan. Each example selects the Rust engine
explicitly.

```{toctree}
:maxdepth: 1

vqe_lih
adapt_vqe_lih
pes_scan
```

The additional tutorials describe specialized workflows. Some still depend on
features awaiting migration; consult the [native coverage](../guide/native_engine.md)
before using them. They are retained as method references during the cutover.

```{toctree}
:maxdepth: 1

vqe_h2
energy_levels
subspace_vqe
mcas_vqe
valqa
bloch_crystals
```
