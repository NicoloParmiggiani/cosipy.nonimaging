## Running simulations to generate IRF and aggregating IRFs

Prerequisites:
- Download the COSI mass model from repository X and place it on your system.
- Edit `bgo.config.yaml` to set the correct path to the downloaded mass model.

1. Change into the directory where you want to generate the simulations.
2. Launch the simulation command, adjusting the number of threads to match your system:

```bash
nohup bc-rsp [path-to-the-config-file]/bgo.config.yaml fullsky_nside32 --ntriggers 200000 --nthreads 200 &
```

After the simulation finishes, use the `aggregate_irf` notebook to aggregate the IRFs following the ACS panel layout.

After aggregating the IRFs, you can use the `visualizeIRF` notebook to plot the expectation maps of the different BGO panels.

