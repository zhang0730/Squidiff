## 📥 安装方式

bash

```
pip install Squidiff
```



------

## 📁 输入数据

模型接受 **h5ad 格式**的单细胞数据文件，需包含：

- 单细胞计数矩阵
- 元数据（如细胞类型、处理条件等）
- （可选）药物化合物信息

------

## ✨ 功能特点

- 预测药物处理后的单细胞转录组
- 预测细胞分化过程
- 预测基因扰动效应

------

## 🛠️ 模型训练

### 基础训练命令：

bash

```
python train_squidiff.py \
  --logger_path LOGGER_FILE_NAME \
  --data_path YOUR_DATASET.h5ad \
  --resume_checkpoint CHECKPOINT_NAME \
  --gene_size 500 \
  --output_dim 500
```



### 包含药物结构的训练示例：

bash

```
python train_squidiff.py \
  --logger_path logger_files/logger_sciplex_random_split_0 \
  --data_path datasets/sci_plex_train_random_split_0.h5ad \
  --resume_checkpoint sciplex_results_random_split_0 \
  --use_drug_structure True \
  --gene_size 200 \
  --output_dim 200 \
  --control_data_path datasets/sci_plex_train_random_split_0_control.h5ad
```



------

## 🔬 模型采样与预测

提供了使用训练好的模型进行预测的 Python 代码示例：

python

```
sampler = sample_squidiff.sampler(
    model_path='simu_results/model.pt',
    gene_size=100,
    output_dim=100,
    use_drug_structure=False
)

test_adata_scrna = sc.read_h5ad('datasets/sc_simu_test.h5ad')
z_sem_scrna = sampler.model.encoder(torch.tensor(test_adata_scrna.X).to('cuda'))

scrnas_pred = sampler.pred(z_sem_scrna, gene_size=test_adata_scrna.shape[1])
```



------

## 🔗 演示与复现

更详细的数据准备、使用方法和下游分析请访问：
https://github.com/siyuh/Squidiff_reproducibility

------

## 📚 引用方式

该项目有两篇关联的《Nature Methods》文章需要引用：

bibtex

```
@article{he2025squidiff,
  title={Squidiff: predicting cellular development and responses to perturbations using a diffusion model},
  author={He, S. and Zhu, Y. and Tavakol, D.N. et al.},
  journal={Nature Methods},
  year={2025},
  doi={https://doi.org/10.1038/s41592-025-02877-y}
}

@article{he2025predicting,
  title={Predicting cellular responses with conditional diffusion models},
  author={He, S. and others},
  journal={Nature Methods},
  year={2025},
  doi={https://doi.org/10.1038/s41592-025-02878-x}
}
```



------

## 📧 联系方式

如有问题可联系：

- 作者：Siyu He（siyuhe@stanford.edu）
- 或通过 GitHub Issues 提问

------

## ✅ 总结

**squidiff** 是一个**先进的单细胞转录组预测工具**，利用扩散模型模拟细胞在不同条件下的基因表达变化，适用于**药物研发、发育生物学和基因功能研究**等领域。其代码已封装成 Python 包，便于安装和使用，并提供了完整的训练、预测和复现指南。