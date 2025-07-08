
## Refusion-HDR (ICCV AIM 2025 Inverse Tone Mapping Challenge) 

Our solution simply use extra data from [[HDRCNN]](https://computergraphics.on.liu.se/hdrcnn/)(TOG 2017) and [[Refusion]](https://github.com/Algolzw/image-restoration-sde)(CVPRW 2023). </sub>

## Dependenices

* OS: Ubuntu 20.04
* nvidia :
	- cuda: 11.7
	- cudnn: 8.5.0
* python3
* pytorch >= 1.13.0
* Python packages: `pip install -r requirements.txt`

## How to use our Code?

### Train
The main code for training is in `codes/config/HDR`.

You can train the model following below bash scripts:

```bash
cd codes/config/HDR

# For single GPU:
python train.py -opt=options/train/refusion.yml

# For distributed training, need to change the gpu_ids in option file
torchrun --nproc_per_node=4 --master_port=4321 train.py -opt=options/train/refusion.yml --launcher pytorch
```

Then the models and training logs will save in `log/HDR_sde/`. 
You can print your log at time by running `tail -f log/HDR_sde/train_HDR_sde_***.log -n 100`.

### Evaluation
To evaluate on any dataset, please modify the benchmark path and model path and run

```bash
cd codes/config/HDR
python run_itm.py -opt_path options/test/refusion.yml -input_dir AIM_TEST -output_dir output -pretrained_path lastest_EMA.pth
```

Pretrained model [here](https://www.dropbox.com/scl/fi/yg44t2i9tgrlsn3c1punc/lastest_EMA.pth?rlkey=fhjb37o34i9yt12337pyed5gi&st=43psqej3&dl=0) on ICCV AIM2025 Inverse Tone Mapping Challenge datasets.

### Some Results
![Refusion](figs/HDR.png)
<div align='center'>Inverse Tone Mapping</div>


## Citations
If this solution helps your research or work, please cite;)
The following are BibTeX references:

```
@article{EKDMU17,
  author       = "Eilertsen, Gabriel and Kronander, Joel, and Denes, Gyorgy and Mantiuk, Rafa\l and Unger, Jonas",
  title        = "HDR image reconstruction from a single 
                  exposure using deep CNNs",
  journal      = "ACM Transactions on Graphics (TOG)",
  number       = "6",
  volume       = "36",
  articleno    = "178",
  year         = "2017"
}

@article{luo2023image,
  title={Image Restoration with Mean-Reverting Stochastic Differential Equations},
  author={Luo, Ziwei and Gustafsson, Fredrik K and Zhao, Zheng and Sj{\"o}lund, Jens and Sch{\"o}n, Thomas B},
  journal={International Conference on Machine Learning},
  year={2023},
  organization={PMLR}
}

@inproceedings{luo2023refusion,
  title={Refusion: Enabling Large-Size Realistic Image Restoration with Latent-Space Diffusion Models},
  author={Luo, Ziwei and Gustafsson, Fredrik K and Zhao, Zheng and Sj{\"o}lund, Jens and Sch{\"o}n, Thomas B},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  pages={1680--1691},
  year={2023}
}
```
