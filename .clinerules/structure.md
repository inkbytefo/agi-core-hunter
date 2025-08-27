# AGI Core Hunter - Project Structure & Organization

## Monorepo Architecture
The project follows a monorepo structure with clear separation between shared code, experiments, and documentation.

## Directory Structure

### Root Level
```
/agi_core_hunter/
├── README.md                 # Main project introduction
├── setup.py                  # Package configuration
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT license
├── .gitignore               # Git ignore patterns
└── test_setup.py            # System verification script
```

### Core Source Code (`src/`)
```
src/
├── __init__.py              # Package initialization
├── agents/                  # Agent implementations
│   ├── __init__.py
│   └── mdl_agent.py        # MDL/β-VAE agent
├── core/                    # Base classes and interfaces
│   ├── __init__.py
│   └── base_agent.py       # Abstract agent interface
├── envs/                    # Environment implementations
│   ├── __init__.py
│   └── grid_world.py       # Grid world navigation
└── utils/                   # Shared utilities
    ├── __init__.py
    └── metrics.py          # Evaluation metrics
```

### Experiments (`experiments/`)
Each experiment follows a standardized structure:
```
experiments/01_mdl_vs_ood/
├── manifest.json           # Experiment configuration
├── train.py               # Training script
├── demo.py                # Quick demo/test
├── eval.ipynb             # Analysis notebook
└── *.md                   # Experiment-specific docs
```

### Documentation (`docs/`)
```
docs/
├── PROJE_TANITIM.md       # Project vision (Turkish)
├── PROJE_TEKNIKMIMARI.md  # Technical architecture (Turkish)
├── PROJECT_STATUS.md      # Current status
├── HOW_TO_USE.md         # Usage guide
├── ROADMAP.md            # Development roadmap
└── theory_cards/         # Principle summary cards
```

### Supporting Directories
- `literature/` - Research papers and references
- `notebooks/` - Jupyter notebooks for analysis
- `scripts/` - Utility scripts and setup tools

## Coding Conventions

### File Naming
- Python files: `snake_case.py`
- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`

### Module Organization
- **Abstract base classes** in `src/core/`
- **Concrete implementations** in respective subdirectories
- **Shared utilities** in `src/utils/`
- **Environment-specific code** stays in experiment directories

### Import Patterns
```python
# Standard library imports first
import json
import os
from pathlib import Path

# Third-party imports
import jax
import jax.numpy as jnp
import numpy as np

# Local imports last
from src.core.base_agent import BaseAgent
from src.utils.metrics import calculate_ood_score
```

### Configuration Management
- All experiments use `manifest.json` for reproducible configuration
- No hardcoded parameters in training scripts
- Environment variables for system-level configuration (JAX settings)

### Documentation Standards
- Docstrings for all public classes and methods
- Type hints using `typing` module and `chex` for JAX arrays
- Bilingual documentation: Turkish for research docs, English for technical implementation