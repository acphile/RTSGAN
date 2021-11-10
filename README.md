# Towards Generating Real-World Time Series Data
A `PyTorch` implementation of RTSGAN model. This is the experiment code for our ICDM 2021 paper "Towards Generating Real-World Time Series Data".

## Abstract
> Time series data generation has drawn increasing attention in recent years. Several generative adversarial network (GAN) based methods have been proposed to tackle the problem usually with the assumption that the targeted time series data are well-formatted and complete. However, real-world time series (RTS) data are far away from this utopia, e.g., long sequences with variable lengths and informative missing data raise intractable challenges for designing powerful generation algorithms. In this paper, we propose a novel generative framework for RTS data - RTSGAN to tackle the aforementioned challenges. RTSGAN first learns an encoder-decoder module which provides a mapping between a time series instance and a fixed-dimension latent vector and then learns a generation module to generate vectors in the same latent space. By combining the generator and the decoder, RTSGAN is able to generate RTS which respect the original feature distributions and the temporal dynamics. To generate time series with missing values, we further equip RTSGAN with an observation embedding layer and a decide-and-generate decoder to better utilize the informative missing patterns. Experiments on the four RTS datasets show that the proposed framework outperforms the previous generation methods in terms of synthetic data utility for downstream classification and prediction tasks.

## Requirements
### Packages
```
pytorch==1.6.0
fastNLP==0.5.5
sklearn==0.23.1
```
### Other git repositories
```
# outside this respository
git clone https://github.com/YerevaNN/mimic3-benchmarks.git
git clone https://github.com/Luoyonghong/Multivariate-Time-Series-Imputation-with-Generative-Adversarial-Networks.git
git clone https://github.com/jsyoon0823/TimeGAN.git
```

## How to Run 
### Complete time series generation
Prepare .pkl data by running `prepare_stock_and_energy.ipynb`

Then train the generative model (take `Stock` dataset as an example) by 
```
python main_stock.py --dataset ./data/stock.pkl --task-name stock
```
The generated data is stored at `../stock_results/stock/data`, and you can evaluate it according to `https://github.com/jsyoon0823/TimeGAN.git`

### Incomplete time series generation
Use Physionet2012 as an example:

First prepare .pkl data by running `prepare_physionet2012.ipynb`

The train the generative model by:
```
python main_2012.py --dataset ./data/physio_data/full2012.pkl --task-name test
```

Then you can examine the quality of synthetic data under the TSTR setting by:
```
python miss2012.py --task-name ../2012_result/test --impute zero
python miss2012_std.py --task-name ../2012_result/test --impute last
```
More arguments for training and evalution can be found in the corresponding files.

For MIMIC-III, you should first have the access to the raw data and follow `https://github.com/YerevaNN/mimic3-benchmarks.git` to generate the corresponding dataset `in-hospital-mortality`. The rest process is similar to above.
 
## Citation
You are more than welcome to cite our paper:
```
@inproceedings{pei2021towards,
  title={Towards Generating Real-World Time Series Data},
  author={Hengzhi Pei, Kan Ren, Yuqing Yang, Chang Liu, Tao Qin, and Dongsheng Li},
  booktitle={Proceedings of the 2021 IEEE International Conference on Data Mining (ICDM), Auckland, New Zealand},
  year={2021}
}
```

