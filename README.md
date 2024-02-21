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
@article{rolinek2020deep,
    title={Deep Graph Matching via Blackbox Differentiation of Combinatorial Solvers},
    author={Michal Rolínek and Paul Swoboda and Dominik Zietlow and Anselm Paulus and Vít Musil and Georg Martius},
    year={2020},
    eprint={2003.11657},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
