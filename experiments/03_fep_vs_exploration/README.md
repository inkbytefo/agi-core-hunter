# Free Energy Principle vs Exploration Experiment

## 🎯 Experiment Overview

This experiment tests the **third AGI principle: Free Energy Principle (FEP)**. It investigates whether agents implementing active inference and variational free energy minimization show better exploration efficiency, adaptation to uncertainty, and surprise minimization compared to other approaches.

### Hypothesis
> Agents with stronger Free Energy Principle implementation (higher epistemic_weight) will show better exploration efficiency, faster adaptation to uncertainty, and more robust performance in dynamic environments.

## 🧪 What This Experiment Tests

### Core Principle: Free Energy Principle & Active Inference
- **Variational Free Energy Minimization**: Agents minimize surprise through perception and action
- **Active Inference**: Actions selected to minimize expected free energy
- **Epistemic vs Pragmatic Trade-off**: Balancing information gain (exploration) with goal achievement
- **Surprise Minimization**: Maintaining homeostasis in uncertain environments

### Key Innovations
1. **FEPAgent**: Implements variational free energy minimization and active inference
2. **ActiveInferenceWorld**: Environment with uncertainty, hidden information, and surprise events
3. **Epistemic Value Calculation**: Measures information gain potential from actions
4. **Surprise-based Adaptation**: Tests response to unexpected environmental changes

## 🚀 Quick Start

### Run the Complete Experiment
```bash
cd experiments/03_fep_vs_exploration
python train.py
```

### Run Quick Demo
```bash
python demo.py
```

### Analyze Results
```bash
jupyter notebook eval.ipynb
```

## 📊 Experiment Components

### Agents Tested
- **FEP_Low** (epistemic_weight=0.1): Minimal exploration drive
- **FEP_Medium** (epistemic_weight=1.0): Balanced exploration/exploitation  
- **FEP_High** (epistemic_weight=3.0): Strong exploration drive
- **MDL_Baseline**: Compression-based baseline (no active inference)

### Environment Features
The `ActiveInferenceWorld` includes:
- **Uncertainty Zones**: Areas with noisy observations testing belief updating
- **Hidden Rewards**: Rewards requiring exploration to discover
- **Dynamic Goals**: Goals that move to test adaptation
- **Surprise Events**: Unexpected environmental changes
- **Exploration Tracking**: Information gain opportunities

### Test Scenarios
1. **High Uncertainty**: Test performance with noisy observations
2. **Exploration Challenge**: Test information gathering efficiency
3. **Surprise Adaptation**: Test response to unexpected changes
4. **Dynamic Goals**: Test adaptation to moving targets

## 📈 Key Metrics

### Active Inference Metrics
- **Variational Free Energy**: Accuracy + Complexity (KL divergence)
- **Expected Free Energy**: Epistemic value + Pragmatic value
- **Epistemic Value**: Information gain from exploratory actions
- **Pragmatic Value**: Goal achievement potential
- **Surprise Minimization**: Negative log-probability of observations

### Performance Metrics
- **Exploration Efficiency**: Information gain per action
- **Adaptation Speed**: Rate of performance improvement
- **Uncertainty Reduction**: Belief updating effectiveness
- **Success Rate**: Task completion under different conditions

## 🔬 Technical Implementation

### FEPAgent Architecture
```
Observation → BeliefNetwork → Posterior Beliefs
              ↓
Beliefs → GenerativeModel → A & B Matrices
          ↓
(Beliefs + Preferences) → PolicyNetwork → Action Selection
          ↓
Action → ExpectedFreeEnergyCalculator → Epistemic & Pragmatic Values
```

### Active Inference Components
1. **GenerativeModel**: Learns P(observation|state) and P(state_t+1|state_t, action)
2. **BeliefNetwork**: Updates posterior beliefs via Bayesian inference
3. **PolicyNetwork**: Selects actions based on expected free energy
4. **ExpectedFreeEnergyCalculator**: Computes information gain and goal value

### Environment Extensions
- **Uncertainty Mapping**: Variable observation noise across locations
- **Hidden Information**: Goals and rewards not immediately visible
- **Surprise Generation**: Random environmental changes
- **Exploration Rewards**: Bonus for visiting new areas

## 📋 Expected Results

### If Hypothesis is Supported
- Higher epistemic_weight → better exploration efficiency
- FEP agents → better adaptation to uncertainty
- Strong correlation between exploration drive and performance
- Better surprise minimization with higher FEP strength

### Key Comparisons
- **FEP_High** should outperform **FEP_Low** on exploration tasks
- **FEP agents** should adapt faster than **MDL_Baseline** to surprises
- **Epistemic value** should correlate with exploration success
- **Variational free energy** should decrease with experience

## 🔧 Configuration

Edit `manifest.json` to customize:
- Agent configurations (epistemic_weight, precision)
- Environment parameters (uncertainty_prob, surprise_threshold)
- Training settings (episodes, batch_size)
- Evaluation scenarios (test types, episodes)

## 📚 Files Overview

- **`train.py`**: Main training script with FEP-specific metrics
- **`demo.py`**: Interactive demonstration of active inference
- **`eval.ipynb`**: Jupyter notebook for result analysis  
- **`manifest.json`**: Experiment configuration
- **`../../src/agents/fep_agent.py`**: FEPAgent implementation
- **`../../src/envs/active_inference_world.py`**: ActiveInferenceWorld environment

## 🎯 Connection to AGI Research

This experiment tests fundamental hypotheses about intelligent behavior:

1. **Active Inference**: Intelligent agents should actively gather information to reduce uncertainty
2. **Surprise Minimization**: Robust intelligence requires maintaining predictable states
3. **Exploration-Exploitation Balance**: Optimal behavior balances information gain with goal achievement
4. **Adaptive Beliefs**: Intelligence requires continuous belief updating based on new evidence

## 🔬 Future Extensions

- Test with more complex belief structures (hierarchical beliefs)
- Add temporal dynamics (planning over multiple time steps)
- Compare with other active inference implementations
- Test on real-world domains with known uncertainty structure

## 📊 Citing This Work

This experiment implements ideas from:
- Friston, K. (2010). The free-energy principle: a unified brain theory?
- Da Costa, L. et al. (2020). Active inference on discrete state-spaces
- Parr, T. & Friston, K. (2019). Generalised free energy and active inference

---

*Part of the AGI Core Hunter project - systematically testing fundamental principles of artificial general intelligence.*