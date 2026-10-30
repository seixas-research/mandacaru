(polarizability)=
# Electric fields and the polarizability

Not yet available. `Mandacaru` has no option to apply a uniform static electric
field, no dipole-moment property, and no polarizability helper; a constructor
keyword such as `electric_field` is rejected. The status and the plan are in
[not yet available](#not-yet-available).

What the calculator returns for a molecule is the energy, the magnetic moment of
a Kohn–Sham run, and (for the supported methods) the forces
([Forces](forces_relaxation.md)); the electron density on the grid is described
in [Visualization](visualization.md).

When the response is added, remember what a response property needs from the
basis: density far from the nuclei. A compact basis underestimates it, and the
`diffuse` option of the PAW-LCAO families exists for that reason
([diffuse functions](#paw-lcao-diffuse)).
