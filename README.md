# Heisenberg exchange and DMI from DFT-calculated supercell total energies

```
pip install totEnJ
```

## Heisenberg exchange

Calculate **Heisenberg exchange** (in-plane and out-of-plane, uniaxial anisotropy) from total energy of magnetic supercells.

<center><img src="https://github.com/user-attachments/assets/32c171bd-507b-4916-8d4a-0f9ca817d598" alt="exchange_total_energy" width="600" /></center>

## DMI

Calculate **Dzyaloshinskii-Moriya interaction coefficients** to arbitrary neighbor from DFT total energy for a linear chain (for now) - many systems will be equivalent however.

- nice-to-have functions:
  - automatically decide what spin spirals to use for a given problem and construct the supercells (and MAGMOM tag)
  - choose along which unit cell vector
