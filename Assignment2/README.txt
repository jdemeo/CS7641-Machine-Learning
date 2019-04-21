Randomized Optimization

- All scripts have been adapted from the ABAGAIL library jython scripts
- Python scripts run appropriate loops to collect essential data for analysis
- Post processing of data and visualization creation were done in a Jupiter notebook

----------------------------------------
- Gamma dataset:
  - gamma_train.txt = training set/validation set for 2-fold cross validation
  - gamma_test.txt = training set/validation set for 2-fold cross validation

- Run the build.xml file to create the .jar file
- Place all scripts and datasets into new bin directory
- Run below files with jython:
  - Gamma dataset:
    - hyperparameter_optimization.py = for optimizing hyperparameters for each algorithm
    - backpropagation_implementation.py = implementation of backdrop network for Gamma datasets network from Supervised Learning assignment
    - gamma_multi_train_test.py = for running optimized randomized hill climbing, simulated annealing, and genetic algorithm on the network
    - helper_functions.py = functions used for cross validation loop
  - Toy optimization problems:
    - knapsack.py = for running knapsack problem and proper iteration loops
    - fourpeaks.py = for running four peaks problem and proper iteration loops
    - travelingsalesman.py = for running traveling salesman problem and proper iteration loops
  - gamma_preprocessing_plots.ipynb = jupyter notebook for processing data collected from experiments and creation of visualizations for report

----------------------------------------

The Jupiter notebook is commented appropriately to explain what each script does. Scripts in notebook indicate where variables should be changed to get the appropriate visualization.