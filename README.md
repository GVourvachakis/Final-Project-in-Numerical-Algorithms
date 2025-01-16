# Final Project on Numerical Algorithms

This repository contains the final project for the Christmas holidays, summarizing key numerical algorithms studied in class with a practical approach. The project focuses on three main topics:

1. **PageRank Algorithm**: Based on the Random Surfer Model, analyzing the Markovian normalized adjacency matrix (inspired by S.Brin and L. Page in 1998 in the paper entitled ["The anatomy of a large-scale hypertextual Web search engine"](https://snap.stanford.edu/class/cs224w-readings/Brin98Anatomy.pdf) ).

2. **Gradient-based Minimizers**: Exploring plain Gradient Descent (GD) and Armijo's backtracking line search method.

3. **Spectral Face Recognition**: Implementing a simple classifier for recognizing known/unknown faces and face/non-face classification by analyzing the "face space" intrinsic dimensionality.

## Repository Overview

### Reports and Presentations

- A `Report.pdf` outlining the methodology and results.

- The `Project_Assignment.pdf` containing the project objectives and guidelines.

- `Beamer_slides.pdf`, a presentation summarizing the project outcomes.

### Project Directories

#### Project_P1: PageRank Algorithm
This part focuses on the PageRank Algorithm, analyzing graph structures and computing rankings using the power method. It also compares the results with NetworkX utilities.

- The `main.py` script serves as the entry point for execution.
- The `pagerank` package including modules for handling graph construction and adjacency matrix operations, the power method for PageRank computation, visual comparison across sample graphs and different p-norms of PageRank, and additional integration of NetworkX's PageRank API.
- Sample adjacency matrices for testing are provided as text files.

#### Project_P2: Gradient-based Minimizers

This section delves into optimization techniques, implementing gradient descent and Armijo's line search to minimize test objective functions from [Simon Fraser University](https://www.sfu.ca/~ssurjano/index.html).
Core modules include:

- `comparison_utils.py`: Tools for comparing optimization methods.

- `line_search.py`: Implements Armijo's backtracking line search.

- `lr_playground.py`: Testing different search distances / learning rates.

- `trial_objectives.py`: Defines objective functions for testing. Rosenbrock (valley-shaped), Matyas (plate-shaped) 1D & 2D Griewank (multimodal).

#### Project_P3: Spectral Face Recognition

The final part develops a spectral face recognition system that classifies analyzing the constructed "face space".

- The `Project_P3.py` script handles the entire execution pipeline.

- Image directories include:

    1. `faces/`: Training images. (36 JPEG)

    2. `testfaces/`: Testing images. (10 JPEG)

    3. `basis/`: Stores the computed eigenfaces.

## How to use

1. Clone the repository
    ```html
    git clone https://github.com/GVourvachakis/Final-Project-Numerical-Algorithms.git
    ```

2. Navigate to the desired project directory (`Project_P1`, `Project_P2`, or `Project_P3`) following the above content description.

3. Ensure that all required Python dependencies are installed using `pip`.

This project provides hands-on experience with
implementing and analyzing the PageRank algorithm using the power method, exploring gradient-based optimization techniques with applications in machine learning, and developing a spectral face recognition system to understand intrinsic dimensionality in image data.
