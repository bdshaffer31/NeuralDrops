# NeuralDrOps

Neural Drop Operators
Modeling droplet evaporation dynamics with Fourier Neural Operators, finite differences and neural ODEs.

## Overview

**NeuralDrOps** is a project focused on simulating droplet evaporation dynamics using neural ordinary differential equations (ODEs) and learned control volumes. This approach combines physical modeling with machine learning techniques to accurately predict the behavior of evaporating droplets.

This was submitted as a final project for UPenn ENM 5310 by Ben Shaffer and Michael Machold.

Please see the report for more details.

## Features

- **Neural ODEs**: Utilizes neural networks to model the time evolution of droplet properties.
- **Learned Control Volumes**: Implements machine learning to define control volumes for improved simulation accuracy.
- **Data Visualization**: Includes tools for visualizing simulation results.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/bdshaffer31/NeuralDrops.git
   cd NeuralDrops

2. **Install dependencies**
    Ensure you have Python installed. Then, install the required packages (TODO):
    ```bash
    pip install -r requirements.txt

## Usage

1. **Configure the model**

    Modify model parameters, architecture and data choices in the configurations in main.py if needed.

    Specify a run directory to save models, metrics, and visualizations

2. **Run the main.py script**

    Execute the main script to start the simulation:

    ```bash
    python main.py

3. **Visualize results**
    Plots are generated automatically in the experiment directory

## Project Structure

    ```bash
    NeuralDrops/
    ├── data/             # Contains datasets used for training and validation
    ├── drop_model/       # Neural network models and related scripts
    ├── experiments/      # Configuration files and results of experiments
    ├── resources/        # Additional resources (images, documentation)
    ├── data_viz.py       # Script for data visualization
    ├── load_data.py      # Data loading and preprocessing
    ├── logger.py         # Manages logging of simulation processes
    ├── main.py           # Main script to run simulations
    ├── networks.py       # Defines neural network architectures
    ├── run.py            # Executes training and evaluation routines
    ├── utils.py          # Utility functions for various tasks
    └── visualize.py      # Additional visualization tools

## License
This project is licensed under the MIT license

## Acknowledgments
Special thanks to the contributors of this project. For a complete list, see the contributors page.