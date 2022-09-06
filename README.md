# DCSA-Net

This repository is the official implementation of the paper "Dynamic Convolution Self-Attention Network for Land Cover Classification in VHR Remote Sensing Images". 

## Abstract

The current deep convolutional neural networks for Very-High-Resolution (VHR) remote sensing image land cover classification often suffer from two challenges. First, the feature maps extracted by network encoders based on vanilla convolution usually contain a lot of redundant information, which easily causes misclassification of land cover. Moreover, these encoders usually require a large number of parameters and high computational costs. Second, as remote sensing images are complex and contain many objects with large-scale variances, it is difficult to use the popular feature fusion modules to improve the representation ability of feature maps. To address the above issues, we propose a dynamic convolution self-attention network (DCSA-Net) for VHR remote sensing image land cover classification. The proposed network has two advantages. On the one hand, we design a lightweight dynamic convolution module (LDCM) by using dynamic convo-lution and a self-attention mechanism. This module can extract more useful image features than vanilla convolution, avoiding the negative effect of useless feature maps on land cover classifi-cation. On the other hand, we design a context information aggregation module (CIAM) with a ladder structure to enlarge the receptive field. This module can aggregate multi-scale contexture information from feature maps with different resolutions using the dense connection. Experiments results show that the proposed DCSA-Net is superior to state-of-the-art networks due to higher accuracy of land cover classification, fewer parameters, and lower computational cost.

## DCSA-Net architecture

![alt text](https://github.com/Julia90/DCSA-Net/blob/main/architecture.PNG?raw=true)


## Results

Our model achieves the following performance on land cover classification:


|  Dataset	| Imp. Surf.	| Building	| Low veg.	| Tree	|  Car	| Mean F1 | 	OA  |	mIoU  |
| --------- |-------------|-----------|-----------|-------|-------|---------|-------|-------|
|  Potsdam	|    93.69    |   96.34 	|   88.05 	| 88.87	| 95.63	|  92.52  |	91.25 |	84.24 |
| Vaihingen |    92.11    |   96.19	  |   83.04 	| 90.31	| 82.39	|  88.81  |	90.58 |	78.93 |


## Reference

If you found this code useful, please cite the following paper:\
(This paper is currently under review. The full publication information will be added.)

```
@article{DCSA-Net,
  title={Dynamic Convolution Self-Attention Network for Land Cover Classification in VHR Remote Sensing Images},
  author={Xuan Wang, Yue Zhang, Tao Lei, Yingbo Wang, Yujie Zhai, and Asoke K. Nandi},
}

```
