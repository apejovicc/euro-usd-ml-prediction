EUR/USD Direction Prediction Project
This repository contains a fully executable Python project for comparing classical machine-learning models, boosting methods, and an LSTM network on the same chronological EUR/USD dataset.

Reproducibility
The experiment is driven by a single configuration file:

config.yaml
The end-to-end pipeline is implemented in:

src/pipelines.py
All model outputs are generated automatically from the shared pipeline and saved in:

results/
Python and library versions
The project was validated with Python 3.11.x and the exact package set listed in requirements.txt.

Environment setup
Create and activate a virtual environment.

Install dependencies:

pip install -r requirements.txt

Running the experiment
From the project root:

python run_experiment.py --fred-key demo

Optional flags:

--short runs the shorter 2-year evaluation window from the configuration.
--config points to a different YAML configuration file.
What the pipeline produces
The execution automatically creates the following artifacts:

raw OHLC prices in data/raw/
engineered feature table in data/processed/
predictions for the test set in results/predictions_<tag>.csv
metric table in results/metrics_<tag>.csv
generalization table in results/generalization_<tag>.csv
optimization comparison in results/optimization_comparison_<tag>.csv
split summary and fold metadata in results/summary_<tag>.json
confusion-matrix figures in results/figures/
Important methodological note
The current study setup is intentionally limited to price/technical features, and the macroeconomic branch is disabled in the configuration to avoid future-information leakage from non-point-in-time macro releases.

Output conventions
The reproducibility seed is defined in config.yaml.
The main experiment uses the same train/test split and shared feature matrix for all models.
LSTM stochastic runs are repeated and summarized by mean and standard deviation.