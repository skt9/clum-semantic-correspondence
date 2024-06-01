# Unsupervised Cycle Consistency Based Deep Graph Matching
This repository contains code for the AAAI 24 paper: **Unsupervised Cycle Consistency Based Deep Graph Matching**

It builds on the code from [Black-Box Deep Graph Matching](https://github.com/martius-lab/blackbox-deep-graph-matching).

##  Solver Download
The QAP solver in the paper can be downloaded from [here](https://github.com/vislearn/libmpopt/tree/iccv2021).


## Training

Training can be done by running the following command:

```
python train.py path/to/your/json/config
```

where ``path/to/your/json/config`` is the path to your configuration file. 

## Troubleshooting
* **There are known issues with the occurence of NaNs in training due to version conflicts with torch_geometric libraries.** Please check your installation of torch_geometric and supporting libraries. 

## Citation

```text
@article{tourani2024clum_aaai,
title={Discrete Cycle-Consistency Based Unsupervised Deep Graph Matching},
volume={38},
url={https://ojs.aaai.org/index.php/AAAI/article/view/28332},
DOI={10.1609/aaai.v38i6.28332},
number={6},
journal={Proceedings of the AAAI Conference on Artificial Intelligence},
author={Tourani, Siddharth and Khan, Muhammad Haris and Rother, Carsten and Savchynskyy, Bogdan}, year={2024}, month={Mar.}, pages={5252-5260} }
```
