# MixedTraining

Official implementation of the KI 2025 paper "Development of Hybrid Artificial Intelligence Training on Real and Synthetic Data"

## Overview

A framework for training neural networks on mixtures of real and synthetic data, comparing two strategies:
- **Simple Mixed**: Training on combined real/synthetic datasets with varying proportions
- **Fine-tuned**: Pretraining on synthetic data, then fine-tuning on real data

## Supported Architectures

- CNN (AlexNet-style)
- MLP
- Vision Transformer (ViT)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py --config conf/classification_cifake.yaml --seed 42
```

## Configuration

YAML configs in `conf/` define:
- Dataset paths and image dimensions
- Model architectures and hyperparameters
- Training settings (epochs, batch size, optimizer)

## Project Structure

```
├── conf/                 # Experiment configurations
├── data/                 # Data loading and management
├── models/
│   ├── networks/         # CNN, MLP, ViT implementations
│   ├── model_builder.py  # Model factory
│   └── model_trainer.py  # Training logic
├── utils/                # Config loading, checkpointing, results saving
├── main.py               # Entry point
└── run_experiment.py     # Experiment runner
```

## Datasets

Supports datasets with real/synthetic splits:
- **CIFAKE** (32×32, 10 classes)
- **DomainNet** (320×320, 60 classes)
- **LEGO** (256×256, 134 classes)