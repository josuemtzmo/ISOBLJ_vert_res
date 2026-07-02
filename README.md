# ISOBLJ_vert_res

This repository hosts the files to reproduce the manuscript titled:
> Sensitivity of ice-shelf basal channel melting to channel depth, ice shelf slope, and vertical model resolution

## Repository Structure

```
ISOBLJ_vert_res/
├── manuscript/     # LaTeX source files and related manuscript materials
├── notebooks/      # Jupyterlab notebooks to reproduce the figures used in the manuscript
├── figures/        # Figures used in the manuscript
├── scripts/        # Script to create the initial conditions of the simulation.
└── README.md       # This file
```

## Getting Started

### The model configuration used in the study can be found at:

https://github.com/josuemtzmo/ISOBLJ

### Reproducing the Manuscript

> **Caveat:** The provided notebooks and scripts are configured for the ARCHER HPC environment or the local BAS cluster. You will need to update file paths for your own system.

1. **Run the mode**. 
2. **Generate Figures**: Navigate to the `notebooks/` directory and run the analysis scripts
3. **Compile Manuscript**: Navigate to the `manuscript/` directory and compile the LaTeX files

