
# Multi-level Critical Point Awareness and Aggregation Network for Point Cloud Normal Estimation

## Requirements

we conduct the experiment in the following setting:

- Ubuntu 18.04 
- python==3.9.7 
- torch==1.9.0+cu111
- matplotlib==3.5.1
- numpy==1.21.5
- tensorboardX==2.2
- scikit_learn==1.0.2
- scipy==1.12.0



## How to use the code


### Data praparation

you need to download PCPNet dataset by
```
python get_data.py
```

and place it in ```./Data/```

you can also download PCPNet and SceneNN dataset from [here](https://drive.google.com/drive/folders/1O606EGHrZaDnlOcH1iQD9GbHEINF2-ox?usp=sharing)

### run (Train + Test on PCPNet):

```
python run.py
```

### Train
you can train your own model by
```
python train.py
```

### Test
our trained model is also provided in ```./log_ablation_noLw/test_GAB_RPE/```
you can simply test it with
```
python test.py
```

### Evaluate
run ```python evaluate.py``` to get the final results.

## Acknowledgement
The code is heavily based on [DeepFit](https://github.com/sitzikbs/DeepFit) , [AdaFit](https://github.com/Runsong123/AdaFit) and [PointFormer](https://github.com/Yi-Jun-Chen/DuPMAM).

If you find our work useful in your research, please cite our paper. And please also cite the DeepFit and AdaFit paper.

```
@inproceedings{chen2022pointformer,
  title={PointFormer: a dual perception attention-based network for point cloud classification},
  author={Chen, Yijun and Yang, Zhulun and Zheng, Xianwei and Chang, Yadong and Li, Xutao},
  booktitle={Proceedings of the Asian Conference on Computer Vision},
  pages={3291--3307},
  year={2022}
}

@article{zhu2021adafit,
  title={AdaFit: Rethinking Learning-based Normal Estimation on Point Clouds},
  author={Zhu, Runsong and Liu, Yuan and Dong, Zhen and Jiang, Tengping and Wang, Yuan and Wang, Wenping and Yang, Bisheng},
  journal={arXiv preprint arXiv:2108.05836},
  year={2021}
}

@article{ben2020deepfit,
  title={DeepFit: 3D Surface Fitting via Neural Network Weighted Least Squares},
  author={Ben-Shabat, Yizhak and Gould, Stephen},
  journal={arXiv preprint arXiv:2003.10826},
  year={2020}
}
```
