# PS-ML: Post-Sketch Machine Learning for Enhanced Frequency Estimation

## Overview

PS-ML (Post-Sketch Machine Learning) is a novel approach for improving item frequency estimation in large-scale data streams using machine learning. It enhances the accuracy of various sketch-based methods without requiring modifications to the online recording module. The project utilizes a two-stage approach—first classifying frequency ranges and then applying regression models—to refine sketch estimates efficiently.

## Features

- **Zero Online Modification**: No changes are required to existing sketch-based recording systems.
- **Post-Sketch Learning**: Enhances estimation accuracy at the query stage using machine learning.
- **Two-Stage Approach**:
  - A **Random Forest Classifier** categorizes frequency ranges.
  - **Neural Network Regression Models** refine predictions within each range.
- **Works with Multiple Sketch Implementations**:
  - Count-Min Sketch (CM)
  - Count-Min Sketch with Conservative Update (CU)
  - Count Sketch (CS)
  - Randomized Counter Sharing (RCS)
  - SSVS and other variants

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/aka334/PS-ML.git
   cd PS-ML
   ```

## Usage

### 1. Running the Classification Model

To classify items into frequency-based bins:

```bash
python classifier.py
```

### 2. Running Regression for Frequency Estimation

For cold items:

```bash
python regressorColdItems.py
```

For hot items:

```bash
python regressorHotItems.py
```
### 3. Testing

You can test the model performance using:

```bash
python testmodel.py
```
### 4. Evaluation

To compute accuracy metrics such as **Mean Absolute Error (MAE)**:

```bash
python calculateMAE.py
```

For a detailed bin-based evaluation:

```bash
python calculateBinMae.py
```

## Experimental Setup

- **Dataset**: Uses **CAIDA** traffic traces from 2015 and 2019.
- **Memory Configurations**: Supports 500kbits, 1Mbits, 2Mbits, and 5Mbits.
- **Hardware**:
  - Intel Core i7-13700 (16 cores)
  - 16GB RAM
  - NVIDIA RTX 3090 (24GB VRAM)


## Contributors

- **Aayush Karki** (University of Kentucky) - [aka334@uky.edu](mailto:aka334@uky.edu)
- **Ahashan Habib Niloy** (University of Kentucky) - [ahashan.niloy@uky.edu](mailto:ahashan.niloy@uky.edu)
- **Mehedi Hasan Munna** (University of Kentucky) - [mehedi.munna@uky.edu](mailto:mehedi.munna@uky.edu)
- **Haibo Wang** (University of Kentucky) - [haibo@ieee.org](mailto:haibo@ieee.org)


