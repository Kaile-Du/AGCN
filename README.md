# AGCN: Augmented Graph Convolutional Network for Lifelong Multi-label Image Recognition
Kaile Du, Fan Lyu, Fuyuan Hu, Linyan Li, Wei Feng, Fenglei Xu, Qiming Fu  
![Alt text](https://github.com/vwyf0305/vwyf0305_picture/blob/master/LML_small.png)  
PyTorch implementation of AGCN, ICME 2022.
## Abstract
The Lifelong Multi-Label (LML) image recognition builds an online class-incremental classifier in a sequential multi-label image recognition data stream. The key challenges of LML image recognition are the construction of label relationships on Partial Labels of training data and the Catastrophic Forgetting on old classes, resulting in poor generalization. To solve the problems, the study proposes an Augmented Graph Convolutional Network (AGCN) model that can construct the label relationships across the sequential recognition tasks and sustain the catastrophic forgetting. First, we build an Augmented Correlation Matrix (ACM) across all seen classes, where the intra-task relationships derive from the hard label statistics while the inter-task relationships leverage both hard and soft labels from data and a constructed expert network. Then, based on the ACM, the proposed AGCN captures label dependencies with a dynamic augmented structure and yields effective class representations. Last, to suppress the forgetting of label dependencies across old tasks, we propose a relationship-preserving loss as a constraint to the construction of label relationships. The proposed method is evaluated using two multi-label image benchmarks and the experimental results show that the proposed method is effective for LML image recognition and can build convincing correlation across tasks even if the labels of previous tasks are missing.
## Requirements
- Python3
- Pytorch (>1.0)
- torchvision (>0.2)
- numpy
- pillow~=6.2.1
- torch_optimizer
- randaugment
- pandas~=1.1.3
## Dataset
Split-COCO and Split-NUS
## Training and Testing
`python3 GCN.py`
## Citation
`@article{du2022agcn,  

  title={AGCN: Augmented Graph Convolutional Network for Lifelong Multi-label Image Recognition},  
  
  author={Du, Kaile and Lyu, Fan and Hu, Fuyuan and Li, Linyan and Feng, Wei and Xu, Fenglei and Fu, Qiming},  
  
  journal={arXiv preprint arXiv:2203.05534},   
  
  year={2022}.  
  
}`
