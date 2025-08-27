# AGI Core Hunter - Technical Stack & Build System

## Core Technology Stack

### Programming Language
- **Python 3.9+** - Standard ecosystem with extensive ML library support

### Deep Learning Framework
- **JAX/Flax** (primary) - Functional programming, easy parallelization with `vmap`/`pmap`, ideal for world models and planning
- **PyTorch** (alternative) - Fallback option for specific use cases

### Simulation & Physics
- **Brax** - JAX-based physics simulation, fast and parallelizable
- **gymnax** - JAX-native RL environments
- **pymunk** - Custom 2D physics engine when needed

### Specialized Libraries
- **Causal Inference**: `DoWhy`, `CausalNex`, `EconML` for causal graph discovery and do-calculus
- **Experiment Tracking**: Weights & Biases (WandB), MLflow for metrics and parameter tracking
- **Scientific Computing**: NumPy, SciPy, matplotlib, seaborn, pandas, plotly

### Development Tools
- **Testing**: pytest for unit tests
- **Code Quality**: black (formatting), isort (imports), mypy (type checking)
- **Notebooks**: Jupyter with ipywidgets for analysis and visualization

## Build System & Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Install project in development mode
pip install -e .

# Install with optional dependencies
pip install -e ".[dev,notebooks]"
```

### Running Experiments
```bash
# Navigate to experiment directory
cd experiments/01_mdl_vs_ood

# Run training with default manifest
python train.py

# Run with custom manifest
python train.py --manifest custom_config.json

# Run demo/test
python demo.py
```

### Development Workflow
```bash
# Run tests
pytest

# Format code
black src/ experiments/
isort src/ experiments/

# Type checking
mypy src/

# Run specific experiment analysis
jupyter notebook experiments/01_mdl_vs_ood/eval.ipynb
```

### JAX Configuration
- **GPU Memory Management**: Uses `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8`
- **Precision**: Defaults to float32 for better GPU performance
- **Device Detection**: Automatic GPU/CPU detection with fallback

## Project Structure Conventions
- Each experiment has its own directory under `experiments/`
- Shared code lives in `src/` with modular organization
- All experiments use `manifest.json` for reproducible configuration
- Training scripts follow standard pattern: load manifest → create agent/env → train → log to WandB