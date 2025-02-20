
# ADL Project - Optimizer Evaluation on KMNIST Dataset

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [How to Run the Code](#how-to-run-the-code)
- [Code Structure](#code-structure)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)

## Overview
This project aims to:
- Train a feedforward neural network on the KMNIST dataset.
- Evaluate the performance of three optimizers (Adam, RMSprop, AdamW) with two learning rates (0.001, 0.0005).
- Use 5-fold cross-validation to compute average accuracy and training time for each optimizer and learning rate combination.
- Visualize the results using a bar chart and line plot.

## Requirements
To run this notebook, you need the following Python libraries installed:
- Python 3.x
- PyTorch (`torch`)
- Torchvision (`torchvision`)
- NumPy (`numpy`)
- Matplotlib (`matplotlib`)
- Scikit-learn (`sklearn`)

You can install the required libraries using pip:

```bash
pip install torch torchvision numpy matplotlib scikit-learn
```

## How to Run the Code
### Open the Notebook:
Open the `ADL.ipynb` file in Jupyter Notebook or Google Colab.

### Install Dependencies:
If running in Google Colab, install the required libraries by running:

```python
!pip install torch torchvision numpy matplotlib scikit-learn
```

### Run the Code:
Execute each cell in the notebook sequentially. The notebook will:
- Load and preprocess the KMNIST dataset.
- Define the feedforward neural network.
- Train and evaluate the model using 5-fold cross-validation for each optimizer and learning rate combination.
- Display the results in a table and a graph.

### View Results:
After running all cells, the notebook will display:
- A table summarizing the average accuracy and training time for each optimizer and learning rate.
- A bar chart and line plot comparing the performance of the optimizers.

## Code Structure
The notebook is structured as follows:

### 1. Imports:
Import necessary libraries (PyTorch, NumPy, Matplotlib, etc.).

### 2. Feedforward Neural Network:
Define the neural network architecture (`FeedForwardNN` class).

### 3. Data Loading and Preprocessing:
Load the KMNIST dataset and apply transformations (normalization).

### 4. Hyperparameters:
Set batch size, number of epochs, learning rates, and optimizers.

### 5. Training and Evaluation:
Perform 5-fold cross-validation:
- Train the model using different optimizers and learning rates.
- Compute accuracy and training time for each fold.

### 6. Results Visualization:
Display the results in a table.
Plot a bar chart and line graph comparing optimizer performance.

## Future Work
Future work may include:
- Testing with additional optimizers or hyperparameter configurations.
- Exploring different neural network architectures.
- Expanding the evaluation to other datasets.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
