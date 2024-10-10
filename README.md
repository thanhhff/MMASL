## MMASL: Action Selection Learning for Weakly Labeled Multi-view and Multi-modal Action Recognition

**Authors:** [Trung Thanh Nguyen](https://scholar.google.com/citations?user=QSV452QAAAAJ), [Yasutomo Kawanishi](https://scholar.google.com/citations?user=Tdfw6WMAAAAJ), [Vijay John](https://scholar.google.co.jp/citations?user=Wv71RXYAAAAJ) ,[Takahiro Komamizu](https://scholar.google.com/citations?user=j4n_V44AAAAJ), [Ichiro Ide](https://scholar.google.com/citations?user=8PXJm98AAAAJ)


## Introduction
This repository contains the implementation of MMASL on the MM-Office dataset.
The source code will be made publicly available at a later date. 
For more details, please contact `nguyent[at]cs.is.i.nagoya-u.ac.jp`.

## Environment

The Python code is developed and tested in the environment specified in `environment.yml`. 
Experiments on the MM-Office dataset were conducted on a single NVIDIA RTX A6000 GPU with 48 GB of GPU memory. 
You can adjust the `batch_size` to accommodate GPUs with smaller memory.


## Dataset

Download the MM-Office dataset [here](https://github.com/nttrd-mdlab/mm-office) and place it in the `dataset/MM-Office` directory.

## Training
To train the model, execute the following command:
```
    bash ./scripts/train_MM_ViT_Transformer.sh
```

## Inference
To perform inference, use the following command:
```
    bash ./scripts/infer_MM_ViT_Transformer.sh
```

## Acknowledgment
This work was partly supported by Japan Society for the Promotion of Science (JSPS) KAKENHI JP21H03519 and JP24H00733. The computation was carried out using the General Projects on the supercomputer "Flow" with the Information Technology Center, Nagoya University.

## Citation

If you find this code useful for your research, please cite the following paper:
```
@inproceedings{nguyen2024MMASL,
    title={Action Selection Learning for Weakly Labeled Multi-view and Multi-modal Action Recognition},
    author={Trung Thanh Nguyen, Yasutomo Kawanishi, Vijay John, Takahiro Komamizu, Ichiro Ide},
    year={2024}
}
```

```
@inproceedings{nguyen2024MultiASL,
      title={Action Selection Learning for Multilabel Multiview Action Recognition},
      author={Nguyen, Trung Thanh and Kawanishi, Yasutomo and Komamizu, Takahiro and Ide, Ichiro},
      booktitle={ACM Multimedia Asia 2024},
      pages={1--7},
      year={2024},
}
```
