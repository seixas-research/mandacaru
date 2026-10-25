# Constrained RPA (cRPA) & Downfolding Workflow

This directory demonstrates the end-to-end workflow to:
1. Perform self-consistent, spin-polarized periodic DFT for correlated and magnetic crystals.
2. Construct spin-resolved Maximally Localized Wannier Functions (MLWFs) with energy window disentanglement.
3. Compute the partially screened Coulomb interaction tensor using the constrained Random Phase Approximation (cRPA).
4. Downfold the electronic structure into an effective low-energy Hamiltonian with double-counting subtraction.
5. Solve the correlated effective model using variational quantum eigensolvers (e.g., ADAPT-VQE or VQE).

## Example: Antiferromagnetic NiO (AFM-II)

The reference script `01_nio_afm_crpa_downfolding.py` applies this workflow to rocksalt NiO in its type-II antiferromagnetic configuration (alternating ferromagnetic (111) sheets).

### Execution

```bash
python examples/crpa/01_nio_afm_crpa_downfolding.py
```

### Adapting the Workflow to Other Materials

The script is organized into modular functions that can be customized for other systems (such as MnO, FeO, CoO, cuprates, or nickelates):

1. **`build_structure()`**:
   Define the cell geometry and set initial magnetic moments via `atoms.set_initial_magnetic_moments(...)` to define the target magnetic ordering (AFM, FM, or non-magnetic).
2. **`run_dft()`**:
   Configure the exchange-correlation functional (`xc`), basis set (`PAW-LCAO`), and Monkhorst-Pack k-point mesh.
3. **`compute_wannier()`**:
   Select the target orbital trials (e.g., `guess={"Ni": "d"}` or `guess={"Cu": ["d", "p"]}`) and provide disentanglement energy windows around the Fermi level.
4. **`downfold_and_screen()`**:
   Specify the active subset of Wannier functions (e.g., $e_g$ or $t_{2g}$ orbitals) and the number of active electrons `num_particles=(n_up, n_down)`. Set `coulomb="isolated"` to prevent periodic self-interaction of localized charges.
5. **`solve_quantum_model()`**:
   Map the downfolded model to fermionic operators and simulate using variational quantum algorithms.
