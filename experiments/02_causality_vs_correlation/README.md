# Causality vs Correlation Experiment

## 🎯 Experiment Overview

This experiment tests the **second AGI principle: Causality**. It investigates whether agents that learn causal relationships adapt better to environmental interventions compared to agents using only correlational learning.

### Hypothesis
> Agents with stronger causal reasoning (higher causal_strength) will show better adaptation to environmental interventions and causal transfer learning.

## 🧪 What This Experiment Tests

### Core Principle: Causality over Correlation
- **Causal Learning**: Agents learn cause-effect relationships between environmental variables
- **Intervention Adaptation**: Testing how agents adapt when these relationships are broken
- **Transfer Learning**: How learning from one causal intervention transfers to others

### Key Innovations
1. **CausalAgent**: Implements Structural Causal Models (SCMs) with do-calculus
2. **CausalGridWorld**: Environment with interventions on causal variables
3. **Intervention Testing**: Systematic testing of adaptation to environmental changes

## 🚀 Quick Start

### Run the Complete Experiment
```bash
cd experiments/02_causality_vs_correlation
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
- **Causal_Weak** (causal_strength=0.1): Minimal causal reasoning
- **Causal_Medium** (causal_strength=1.0): Moderate causal reasoning  
- **Causal_Strong** (causal_strength=5.0): Strong causal reasoning
- **MDL_Baseline**: Compression-based baseline (no causal reasoning)

### Environment Variables
The `CausalGridWorld` includes three key causal variables:
- **wind_strength**: Affects movement reliability
- **visibility**: Affects observation quality
- **goal_stability**: Affects goal position stability

### Intervention Types
1. **Strong Wind**: Forces high wind conditions (movement blocked)
2. **Low Visibility**: Forces poor visibility (noisy observations)
3. **Unstable Goal**: Forces goal position instability
4. **Multiple Interventions**: Simultaneous interventions

## 📈 Key Metrics

### Intervention Adaptation
- **Success Rate**: Task completion under intervention
- **Adaptation Speed**: How quickly performance improves
- **Early vs Late Performance**: Learning curve analysis

### Causal Transfer Learning
- **Transfer Efficiency**: Performance on new intervention after learning another
- **Zero-shot Transfer**: Immediate performance on unseen interventions
- **Few-shot Transfer**: Performance after minimal training

### Causal Discovery
- **Graph Accuracy**: How well learned causal structure matches ground truth
- **Intervention Prediction**: Accuracy of predicting intervention outcomes

## 🔬 Technical Implementation

### CausalAgent Architecture
```
Observation → CausalEncoder → Causal Variables
                           ↓
Causal Variables → CausalGraphNetwork → Adjacency Matrix
                           ↓
(Variables + Graph) → CausalPolicy → Action
```

### Causal Learning Components
1. **CausalEncoder**: Extracts causal variables from observations
2. **CausalGraphNetwork**: Learns causal relationships (adjacency matrix)
3. **InterventionPredictor**: Predicts outcomes under do(X=x) interventions
4. **CausalPolicyNetwork**: Uses causal reasoning for action selection

### Environment Extensions
- **Intervention Support**: Can apply do(variable=value) interventions
- **Causal Variable Tracking**: Monitors wind, visibility, goal stability
- **Intervention History**: Tracks adaptation over time

## 📋 Expected Results

### If Hypothesis is Supported
- Stronger causal reasoning → better intervention adaptation
- CausalAgent > MDLAgent on intervention tasks
- Good transfer between related interventions

### Key Comparisons
- **Causal_Strong** should outperform **Causal_Weak** and **MDL_Baseline**
- **Transfer efficiency** should be highest for strongest causal agents
- **Adaptation speed** should correlate with causal_strength parameter

## 🔧 Configuration

Edit `manifest.json` to customize:
- Training episodes and evaluation frequency
- Agent configurations (causal_strength, learning_rate)
- Environment parameters (intervention_probability)
- Evaluation scenarios (intervention types, episodes)

## 📚 Files Overview

- **`train.py`**: Main training script with intervention testing
- **`demo.py`**: Interactive demonstration of causal reasoning
- **`eval.ipynb`**: Jupyter notebook for result analysis
- **`manifest.json`**: Experiment configuration
- **`../../src/agents/causal_agent.py`**: CausalAgent implementation
- **`../../src/envs/causal_grid_world.py`**: CausalGridWorld environment

## 🎯 Connection to AGI Research

This experiment tests a fundamental hypothesis in AGI research: that **causal reasoning is superior to correlational learning** for robust intelligence. Key connections:

1. **Robustness**: Causal models should be more robust to distribution shifts
2. **Transfer Learning**: Causal knowledge should transfer better across domains
3. **Intervention Reasoning**: AGI systems need to reason about actions and consequences
4. **Scientific Discovery**: Causal reasoning is essential for understanding the world

## 🔬 Future Extensions

- Test with more complex causal structures
- Add temporal causality (time-delayed effects)
- Compare with other causal discovery methods
- Test on real-world domains with known causal structure

## 📊 Citing This Work

This experiment implements ideas from:
- Pearl, J. (2009). Causality: Models, Reasoning and Inference
- Schölkopf, B. et al. (2021). Toward Causal Representation Learning
- Bengio, Y. et al. (2019). Meta-Learning Framework for Causal Discovery

---

*Part of the AGI Core Hunter project - systematically testing fundamental principles of artificial general intelligence.*