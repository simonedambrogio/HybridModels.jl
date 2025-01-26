# HybridModels.jl

> ⚠️ **Development Status**: This package is in early development and should be considered alpha software. It is not yet ready for production use. Expect frequent breaking changes, incomplete features, and potential bugs. We welcome feedback and contributions to help improve the package.

A Julia package for implementing hybrid models that combine knowledge-driven and data-driven approaches in cognitive neuroscience and behavioral modeling.

## Overview

HybridModels.jl provides an efficient and simple-to-use framework for developing hybrid models that integrate traditional computational models with deep learning approaches. This package is particularly useful for researchers in cognitive neuroscience and psychology who want to leverage both domain knowledge and data-driven methods to understand complex behavioral patterns.

## Motivation

Traditional approaches in cognitive neuroscience often rely on either purely knowledge-driven models (based on theoretical principles) or purely data-driven approaches (like deep learning). Each has its limitations:

- Knowledge-driven models may be too constrained to capture complex behavioral patterns
- Pure data-driven approaches may require excessive data and lack interpretability

HybridModels.jl bridges this gap by allowing researchers to:
1. Incorporate established cognitive principles as knowledge-driven components
2. Use neural networks to model complex, hard-to-define aspects of cognition
3. Create more expressive yet interpretable models
4. Achieve better data efficiency compared to purely data-driven approaches

## Features

- Flexible architecture for combining knowledge-driven and data-driven components
- Implementation of common cognitive modeling components (e.g., UCB algorithm)
- Neural network integration for learning complex information sampling strategies
- Tools for model training, evaluation, and analysis
- Efficient implementation optimized for Julia

## Installation

> Note: As this package is in early development, it is not yet registered in Julia's General registry.

```julia
using Pkg
Pkg.add(url="https://github.com/yourusername/HybridModels.jl")
```

Known Limitations:
- The API is subject to change without notice during this early phase
- Some features may be incomplete or unstable
- Documentation is still being developed
- Test coverage is limited

## Quick Start

```julia
using HybridModels

# Create a hybrid model
model = HybridModel(
    knowledge_component = UCBComponent(),
    data_component = NeuralComponent(hidden_layers=[64, 32])
)

# Train the model
fit!(model, data)

# Make predictions
predictions = predict(model, new_data)
```
