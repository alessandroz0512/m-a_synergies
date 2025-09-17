# M&A Synergy Monte Carlo Simulation

This repository contains a Python simulation framework to evaluate potential M&A synergies using Monte Carlo methods.

## Overview
The simulation models a hypothetical M&A deal by combining:

- Revenue synergies with ramp-up assumptions
- Cost synergies with annual run-rate
- Integration costs modeled as lognormal distributions
- Correlation between revenue and cost synergies

The simulation generates **10,000+ scenarios**, calculates the NPV for each, and provides probabilistic metrics such as:

- Mean and median NPV
- Probability of positive NPV
- Value at Risk (5th percentile)

Histograms and figures visualize the distribution of possible outcomes.

## Features

- Monte Carlo simulation of uncertain synergy components
- Incorporates correlated variables for revenue and costs
- Stochastic modeling of integration costs
- Generates CSV output with detailed simulation results
- Produces high-resolution histograms for analysis

## Requirements

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- SciPy

## Usage

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mna-synergy-simulation.git
