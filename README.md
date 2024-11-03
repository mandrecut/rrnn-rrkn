# rrnn-rrkn Residual Random Neural and Kernel Networks

This is the repository for the paper: https://arxiv.org/abs/2410.19987

## Abstract

The single-layer feedforward neural network with random weights is a recurring motif in the neural networks literature. The advantage of these networks is their simplified training, which reduces to solving a ridge-regression problem. A general assumption is that these networks require a large number of hidden neurons relative to the dimensionality of the data samples, in order to achieve good classification accuracy. Contrary to this assumption, here we show that one can obtain good classification results even if the number of hidden neurons has the same order of magnitude as the dimensionality of the data samples, if this dimensionality is reasonably high. 
Inspired by this result, we also develop an efficient iterative residual training method for such random neural networks, and we extend the algorithm to the least-squares kernel version of the neural network model. Moreover, we also describe an encryption (obfuscation) method which can be used to protect both the data and the resulted network model. 
