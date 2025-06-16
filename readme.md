# A Joint Continuous-Discrete Latent Variable Parametrization Framework for Geological Model Calibration Using Conditional Variational Autoencoders

This repository contains the implementation of a novel framework for geological model calibration that combines continuous and discrete latent variables using Conditional Variational Autoencoders (cVAEs). This hybrid parametrization framework enables controlled generation of complex geological realizations, supports automated scenario selection during
inverse modeling, and enhances interpretability by incorporating prior geological knowledge
directly into the model calibration process

## 📋 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Data Requirements](#data-requirements)
- [Model Architecture](#model-architecture)
- [Two-Phase Flow Simulation](#two-phase-flow-simulation)
- [Results and Visualization](#results-and-visualization)

## 🎯 Overview

This work presents a joint continuous-discrete latent variable parametrization framework that leverages Conditional Variational Autoencoders to efficiently represent geological models. The framework:

- **Combines continuous and discrete latent variables** for flexible geological representation
- **Uses conditional VAE architecture** to handle complex geological scenarios
- **Integrates with two-phase flow simulation** for history matching
- **Provides efficient gradient-based optimization** for model calibration
- **Supports uncertainty quantification** in reservoir modeling

## 📁 Repository Structure

```
├── conditionalVAE/              # Conditional VAE model implementation
│   ├── models.py               # cVAE model architecture
│   ├── training.py             # Training utilities and loss functions
│   └── __init__.py
├── utils/                      # Utility functions
│   ├── dataloaders.py          # Data loading and preprocessing
│   ├── load_model.py           # Model loading utilities
│   └── __init__.py
├── viz/                        # Visualization tools
│   ├── visualize.py            # Training visualization
│   ├── latent_traversals.py    # Latent space exploration
│   └── __init__.py
├── two_phase_flow/             # Two-phase flow simulation (MATLAB)
│   ├── two_phase_forward.m     # Forward simulation
│   ├── two_phase_forward_batch.m # Batch simulation
│   ├── two_phase_forward_with_adjoint.m # Adjoint gradient computation
│   ├── job.slurm               # SLURM job script for HPC
│   ├── test_nogpu.slurm        # SLURM script without GPU
│   └── *.mat                   # Well configuration files
├── train_conditionalVAE.py     # Main training script
├── model_calibration_cVAE_nonadjoint.py # Model calibration script
├── two_phase_flow_viz.py       # Flow simulation visualization
└── README.md                   # This file
```

## 🔧 Dependencies

### Python Dependencies
- **PyTorch** (>= 1.8.0) - Deep learning framework
- **NumPy** (>= 1.19.0) - Numerical computing
- **SciPy** (>= 1.5.0) - Scientific computing and optimization
- **Matplotlib** (>= 3.3.0) - Plotting and visualization
- **scikit-image** (>= 0.17.0) - Image processing
- **torchvision** (>= 0.9.0) - Computer vision utilities
- **matlab.engine** - MATLAB Engine API for Python

### MATLAB Dependencies
- **MATLAB R2022b** or later
- **MRST (MATLAB Reservoir Simulation Toolbox)** - Reservoir simulation framework
- **MATLAB Engine API for Python** - For Python-MATLAB integration

### System Requirements
- **Memory**: 16GB RAM recommended for training
- **GPU**: CUDA-compatible GPU recommended (optional but highly recommended)
- **Storage**: Sufficient space for geological datasets and model checkpoints

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Joint-Discrete-Continous-Parametrization
   ```

2. **Install Python dependencies:**
   ```bash
   pip install torch torchvision numpy scipy matplotlib scikit-image
   ```

3. **Install MATLAB Engine API for Python:**
   ```bash
   cd "matlabroot/extern/engines/python"
   python setup.py install
   ```
   Replace `matlabroot` with your MATLAB installation path.

4. **Install MRST:**
   - Download MRST from [https://www.sintef.no/projectweb/mrst/](https://www.sintef.no/projectweb/mrst/)
   - Add MRST to your MATLAB path

5. **Set up data directory:**
   ```bash
   mkdir ../geo_dataset
   # Place your geological datasets in this directory
   ```

## 📖 Usage

### Step 1: Train the Conditional VAE Model

```bash
python train_conditionalVAE.py
```

**Key Parameters:**
- `batch_size`: 64 (default)
- `lr`: 5e-4 (learning rate)
- `epochs`: 30 (training epochs)
- `latent_spec`: {'cont': 10, 'cond': 4} (10 continuous + 4 discrete latent variables)

**Output:**
- Trained model saved as `model.pt`
- Training logs in `Experiments/model_training/straight_generative_scenario/conditional_VAE/`

### Step 2: Run Model Calibration

**⚠️ Important Note**: Model calibration uses numerical gradients which require multiple forward simulations and can be computationally intensive. We **highly recommend** running on a High-Performance Computing (HPC) cluster.

#### Local Execution (for small tests):
```bash
python model_calibration_cVAE_nonadjoint.py
```

#### HPC Execution (recommended):
```bash
# Submit to SLURM queue
sbatch two_phase_flow/job.slurm

# Or for non-GPU nodes
sbatch two_phase_flow/test_nogpu.slurm
```

**Key Features:**
- **History matching** using two-phase flow simulation
- **Gradient-based optimization** with numerical gradients
- **One-hot penalty** for discrete latent variables
- **Visualization** of optimization progress
- **Batch processing** for multiple test cases

**Output:**
- Calibrated models for each test case
- Optimization history and convergence plots
- History matching results
- Saturation and pressure field comparisons

## 📊 Data Requirements

The framework expects geological datasets in the following format:

- **Training Data**: PyTorch tensors (.pt files) or pickle files (.pkl)
- **Image Size**: 64×64 or 32×32 grayscale images
- **Data Structure**: 
  - Images: Normalized permeability fields [0, 1]
  - Labels: Discrete geological scenario indicators
- **Dataset Location**: `../geo_dataset/`

## 🏗️ Model Architecture

### Conditional VAE Structure

The cVAE model consists of:

1. **Encoder Network:**
   - Convolutional layers for feature extraction
   - Latent space parameterization (mean, log-variance for continuous; alpha for discrete)

2. **Latent Space:**
   - **Continuous variables**: 10-dimensional normal distribution
   - **Discrete variables**: 4-dimensional categorical distribution (geological scenarios)

3. **Decoder Network:**
   - Transposed convolutions for image reconstruction
   - Sigmoid activation for output normalization

### Training Strategy

- **Loss Function**: ELBO (Evidence Lower BOund) with capacity annealing
- **Capacity Schedule**: `[0.0, 0, 25000, 10]` for disentanglement
- **Optimizer**: Adam with learning rate 5e-4
- **Regularization**: One-hot penalty for discrete variables

## 🌊 Two-Phase Flow Simulation

The framework integrates with MRST for reservoir simulation:

### Simulation Features
- **Forward Simulation**: Two-phase oil-water flow
- **Adjoint Gradients**: Efficient gradient computation
- **Batch Processing**: Parallel simulation for multiple realizations
- **Well Configurations**: Support for injection and production wells

### Key Functions
- `two_phase_forward.m`: Forward simulation
- `two_phase_forward_batch.m`: Batch simulation
- `two_phase_forward_with_adjoint.m`: Adjoint gradient computation
- `two_phase_forward_details.m`: Detailed simulation results

## 📈 Results and Visualization

### Training Visualization
- Real-time training progress
- Latent space traversals
- Generated sample quality assessment
- Loss convergence plots

### Calibration Results
- **History Matching Plots**: Well production rates and pressure
- **Field Comparisons**: Saturation and pressure distributions
- **Optimization Progress**: Objective function convergence
- **Parameter Evolution**: Latent variable trajectories

### Output Files
- `calibrator_monitor*.png`: Optimization progress visualization
- `optimization_process_*.png`: Convergence plots
- `history_matching_case*.png`: Production history matching
- `saturation_pressure_case*.png`: Field comparisons

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

For questions and support, please contact [your-email@institution.edu]

---

## ⚠️ Important Notes

### MATLAB Licensing
**Note**: While MRST (MATLAB Reservoir Simulation Toolbox) is open source, **MATLAB itself requires a commercial license**. You will need to obtain a valid MATLAB license to run the two-phase flow simulation components of this framework. Please ensure you have proper licensing before attempting to run the model calibration scripts.

### Computational Requirements
- **Model Calibration**: The calibration process uses numerical gradients which require multiple forward simulations per optimization step. This can be computationally intensive and time-consuming.
- **HPC Recommendation**: We strongly recommend running the model calibration on a High-Performance Computing (HPC) cluster using the provided SLURM scripts in the `two_phase_flow/` directory.
- **Resource Requirements**: Depending on the problem size, calibration may require significant computational resources and wall time.



