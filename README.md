# MG-WFBP: Merging Gradients Wisely for Efficient Communication in Distributed Deep Learning 
## Introduction
This repository contains the codes of the MG-WFBP (Merged-Gradient Wait-Free BackPropagation) paper submitted to *IEEE TPDS*. This version works well with PyTorch, and a preliminary version was presented at the conference *IEEE INFOCOM 2019*, which was originally implemented on B-Caffe: [https://github.com/hclhkbu/B-Caffe](https://github.com/hclhkbu/B-Caffe). As PyTorch becomes much popular than Caffe, you are recommended to use this repository with the MG-WFBP algorithm.

## Installation
### Prerequisites
- Python 2 or 3
- PyTorch-0.4.+
- [OpenMPI-4.0.+](https://www-lb.open-mpi.org/software/ompi/v4.0/)
- [Horovod-0.14.+](https://github.com/horovod/horovod)

### Quick Start
```
git clone https://github.com/HKBU-HPML/MG-WFBP.git
cd MG-WFBP
pip install -r requirements.txt
dnn=resnet20 nworkers=4 ./dist_mpi.sh
```
Assume that you have 4 GPUs on a single node and everything works well, you will see that there are 4 workers running at a single node training the ResNet-20 model with the Cifar-10 data set using the MG-WFBP algorithm.

## Papers
- S. Shi, X.-W. Chu, and B. Li, “MG-WFBP: Merging Gradients Wisely for Efficient Communication in Distributed Deep Learning,” Under review (Extension of the following conference version).
- S. Shi, X.-W. Chu, and B. Li, “MG-WFBP: Efficient Data Communication for Distributed Synchronous SGD Algorithms,” IEEE INFOCOM 2019, Paris, France, May 2019. [PDF](https://arxiv.org/pdf/1811.11141)

## Referred Models
- Deep speech: [https://github.com/SeanNaren/deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch)
- PyTorch examples: [https://github.com/pytorch/examples](https://github.com/pytorch/examples)
