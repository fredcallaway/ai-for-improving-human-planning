# AI for improving human planning

This repository contains data and analysis code for "Leveraging Artificial Intelligence to Improve People’s Planning Strategies" submitted to PNAS in March 2020.

The code for Experiment 1 (with a modification that allows you to select your own condition) is in `demo-experiment`. You can view the demo experiment yourself at https://cocosci.dreamhosters.com/webexpt/improving-planning/

Note that in both the data and the analysis, experiments 4 and 5 are swapped. That is, the data for "Experiment 4" in the paper is in `data/5` and is analyzed by setting `EXPERIMENT = 5` in `analysis/main.ipynb`. All experiments but Experiments 6 and 7 are analyzed in this notebook, with 6 and 7 analyzed by `exp6.ipynb` and `exp7.ipynb` respectively (due to larger differences in the required code).

