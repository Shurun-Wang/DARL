[//]: # (# Dual-Agent Adversarial Reinforcement Learning &#40;DARL&#41; Framework for EEG-based Hyperparameter Optimization)
# DARL for HPO

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-orange)](https://pytorch.org/)

Welcome to the official repository for the DRAL framework. Under Review.

### ✨ Key Features
* **Standardized Data Processing Pipeline**
* **Comprehensive Model Zoo**
* **Intelligent Architecture Search**
* **Physiological Explainability (XAI)**

---

## 📑 Table of Contents
1. [Installation](#-installation)
2. [Data Preparation](#-data-preparation)
3. [Model Zoo](#-model-zoo)
4. [Quick Start](#-quick-start)
5. [Repository Structure](#-repository-structure)
6. [License](#-license)

---

## ⚙️ Installation

We highly recommend using Anaconda to create an isolated virtual environment for this project.

```bash
# 1. Clone the repository
git clone https://github.com/Shurun-Wang/DARL.git
cd DARL

# 2. Create and activate the conda environment
conda create -n darl_env python=3.11
conda activate darl_env

# 3. Install required dependencies
pip install -r requirements.txt
````

*(Note: Ensure you install the appropriate PyTorch version for your CUDA toolkit.)*

-----

## 📊 Data Preparation

This project involves the processing of multiple physiological datasets. To ensure full reproducibility, we provide comprehensive preprocessing guidelines.

### 1\. Download Datasets

Please download the raw EEG datasets from their official sources and place them in the `data/[dataset_name]/raw_data` directory. The framework currently supports:

  * **ADHD**: Attention-deficit hyperactivity disorder dataset.
  * **MDD**: Major depressive disorder dataset.
  * **SCH**: Schizophrenia dataset.
  * **IDD**: Intellectual and developmental disorder dataset.

### 2\. Run Preprocessing

Taking the preprocessing of the ADHD dataset as an example:
```bash
python preprocessing/adhd/data_process.py
```
The processed and segmented data will be automatically saved to the `data/adhd/processed_data/` directory.

```bash
python preprocessing/adhd/json_process.py
```
The data json will be automatically saved to the `preprocessing/adhd/json_1/` directory.

-----

## 🧠 Model Zoo

To evaluate the generalization capabilities of our HPO framework, we implemented various architectural paradigms. You can easily switch between them using `.yaml` configuration files:

| Model Name | Architecture Type | Config Reference |
| :--- | :--- |:------------------------------|
| **OhCNN** | 1D-CNN | `scripts/OhCNN/original.yaml` |
| **DeprNet** | 2D-CNN | `scripts/DeprNet/original.yaml`        | 
| **SzHNN** | 1D-CNN + LSTM | `scripts/SzHNN/original.yaml`          | 
| **MBSzEEGNet** | Multi-Branch CNN | `scripts/MBSzEEGNet/original.yaml`     | 
| **STGEFormer**| Spatial-Temporal Transformer | `scripts/STGEFormer/original.yaml`     | 

> **📝 Note:** For detailed information regarding the predefined discrete hyperparameter search spaces (e.g., kernel sizes, hidden dimensions, attention heads), please refer to `optimizer/search_space.py`.

-----

## 🚀 Quick Start

### 1\. DARL Optimization

Execute the proposed DARL algorithm:

```bash
python main_hpc_finder.py --dataset 'adhd' --model_name 'OhCNN' --search_trials 3000
python main_top3_config.py --dataset 'adhd' --model_name 'OhCNN'
```

### 2\. Final Model Training and Evaluation

Once the optimal hyperparameter configurations are identified, comprehensively evaluate them using a subject-independent 5-fold cross-validation strategy:

```bash
python main_model_train.py --dataset 'adhd' --model_name 'OhCNN' --epochs 100 --batch_size 64
```

### 3\. Interpretability & Visualization

Generate global spatial attention topoplots to extract physiological insights using Integrated Gradients:

```bash
python main_plot_topo.py --dataset 'adhd' --model_name 'OhCNN'
```

*This will generate PDF files (e.g., `Topomap_SzHNN_Global.pdf`) illustrating the spatial attention differences between Healthy Controls (HC) and Patients.*


-----

## 📁 Repository Structure

```text
├── data/                    # Data directory
│   ├── adhd/                
│   │    ├──processed_data/  # Preprocessed and segmented data
│   │    ├──raw_data/        # Original downloaded datasets
│   ├── mdd/ 
│   ├── sch/      
│   ├── idd/                    
├── models/                  # Deep learning model definitions
│   ├── get_model.py         
│   ├── complexity.py        
│   ├── DeprNet.py           
│   ├── MBSzEEGNet.py
│   ├── OhCNN.py
│   ├── STGEFormer.py
│   └── SzHNN.py
├── scripts/                 # Utility scripts for HPC
├── results/                 # XAI results
├── run/                     # Utility scripts for running this project
├── optimizer/               # DARL Optimization framework
│   ├── ahpo.py              
│   └── search_space/  
├── preprocessing/           # Data preprocessing and data spilt     
├── main_hpc_finder.py       # Main entry for the architecture search phase
├── main_top3_config.py      # Main entry for selecting the best HPC
├── main_model_train.py      # Main entry for the 5-fold CV evaluation phase
├── main_plot_topo.py        # Main entry for XAI phase
├── utils.py                 # tools
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

-----

## 📜 License

This project is licensed under the [MIT License](https://www.google.com/search?q=LICENSE).


```
```
