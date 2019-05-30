# Benchmarking of CD8+ T-cell epitope prediction methods

This repository contains the data and code used in benchmarking of the CD8+ T-cell epitope prediction methods.
More details are in the manuscript (currently under preparation) 

- data/0.complete_data.zip contains all data needed to do the performance evaluations. It includes all peptides, prediction scores from each method, immunogenicity category, T-cell response scores and other parameters.
- benchmarking.py takes the above data file and generates the result files and plots. The configurations for running the script are in config.py
- The result files and plots are in the directory results/

# Requirements (the following python packages are required):
- scikit-learn
- matplotlib
- numpy
- time
