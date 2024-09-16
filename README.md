# Heisenberg exchange and DMI from DFT-calculated supercell total energies

<i>**Tools for preparing magnetic supercells and extracting (anisotropic) exchange interaction and DMI from their total energies (calculated for instance by DFT).**</i>

```
pip install totEnJ
```

If you find this package useful, please cite [L. Vojáček*, J. M. Dueñas* _et al._, Nano Letters (2024)](https://pubs.acs.org/doi/10.1021/acs.nanolett.4c03029).

## Heisenberg exchange

Calculate **Heisenberg exchange** (in-plane and out-of-plane, uniaxial anisotropy) from total energy of magnetic supercells.

<center><img src="https://github.com/user-attachments/assets/32c171bd-507b-4916-8d4a-0f9ca817d598" alt="exchange_total_energy" width="600" /></center>

## DMI

Calculate **Dzyaloshinskii-Moriya interaction coefficients** to arbitrary neighbor from DFT total energy for a linear chain (for now) - many systems will be equivalent however.

- nice-to-have functions:
  - automatically decide what spin spirals to use for a given problem and construct the supercells (and MAGMOM tag)
  - choose along which unit cell vector
