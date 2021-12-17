# PASS: Performance-Adaptive Sampling Strategy Towards Fast and Accurate Graph Neural Networks

PASS is a neighborhood sampler for graph neural network models.
PASS samples neighbors informative for a target task by optimizing a sampling policy directly towards task performance.

You can see our KDD 2021 paper ["Performance-Adaptive Sampling Strategy Towards Fast and Accurate Graph Neural Networks"](https://minjiyoon.xyz/Paper/PASS.pdf) for more details.
This implementation is based on Pytorch.

## Requirement

Use the package manager pip to install requirements:

```bash
pip install -r requirements.txt
```

## Dataset

We use open-source dataset, [GNN-benchmark](https://github.com/shchur/gnn-benchmark), for our experiments.
Our code reads npz-format graph datasets.


## Usage

In **args.py**, you can find a list of hyperparameters.
Some of them are related to Neural Network training, some are related to GNN structure, and the others are related to our sampling strategies.
You can find descriptions of hyperparameters in **args.py** file.

Here is the example command to run PASS.
```bash
python test.py --dataset cora --sample_num 5
```
Once you download all npz-format datasets in **run.sh** into ./Data/ directory, you can simply run **run.sh** to test all datasets with different sampling numbers.


## Citation

Please consider citing the following paper when using our code for your application.
```bash
@inproceedings{yoon2021performance,
  title={Performance-Adaptive Sampling Strategy Towards Fast and Accurate Graph Neural Networks},
  author={Yoon, Minji and Gervet, Th{\'e}ophile and Shi, Baoxu and Niu, Sufeng and He, Qi and Yang, Jaewon},
  booktitle={Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery \& Data Mining},
  pages={2046--2056},
  year={2021}
}
```
