# 12x12 D=4 Example: (pi/2,pi/2) Stripe Regime

This example bundle matches the main-text stripe regime parameters:

- `t = 1.0`
- `t2 = 0.3`
- `U = 14.0`
- `Jk = -4.0`  (`J_H = 4t` in the paper notation)
- quarter filling on `12x12`: `ElectronNum = 72`
- `InitState = "stripe_pi2pi2"`

Files:

- `physics_12x12_pi2pi2.json`
- `simple_update_d4.json`
- `vmc_sr_d4.json`
- `mc_measure_d4.json`

Example commands:

```bash
./peps_kondo_square_simple_update \
  src_peps_kondo_single_layer/params/examples_12x12_pi2pi2_d4/physics_12x12_pi2pi2.json \
  src_peps_kondo_single_layer/params/examples_12x12_pi2pi2_d4/simple_update_d4.json

mpirun -n <N> ./peps_kondo_square_vmc_optimize \
  src_peps_kondo_single_layer/params/examples_12x12_pi2pi2_d4/physics_12x12_pi2pi2.json \
  src_peps_kondo_single_layer/params/examples_12x12_pi2pi2_d4/vmc_sr_d4.json

mpirun -n <N> ./peps_kondo_square_mc_measure \
  src_peps_kondo_single_layer/params/examples_12x12_pi2pi2_d4/physics_12x12_pi2pi2.json \
  src_peps_kondo_single_layer/params/examples_12x12_pi2pi2_d4/mc_measure_d4.json
```

Practical note for MPI runs:

- After Simple Update, only `tpsfinal/configuration0` is dumped by default.
- For multi-rank VMC/measure jobs, you can duplicate rank-0's configuration to avoid
  random-sector fallback and rescue messages:

```bash
for r in $(seq 1 $((N - 1))); do
  cp tpsfinal/configuration0 "tpsfinal/configuration${r}"
  cp tpsfinal/configuration0.shape "tpsfinal/configuration${r}.shape"
done
```

These are starter parameters, not tuned production settings. Increase VMC iterations,
BMPS dimensions, and MC sample counts after the first cluster smoke run if needed.
