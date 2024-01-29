# Imitation Learning from Observation with Automatic Discount Scheduling

[\[Website\]](https://il-ads.github.io/) [\[arXiv\]](https://arxiv.org/abs/2310.07433) [\[OpenReview\]](https://openreview.net/forum?id=pPJTQYOpNI)


## Instructions

### Environment Setup
- Install [Mujoco](http://www.mujoco.org/) based on the instructions given [here](https://github.com/facebookresearch/drqv2).

- Install the following libraries:
  ```
  sudo apt update
  sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
  ```

- Install other dependencies:
  ```
  conda env create -f conda_env.yml
  conda activate ads
  ```
  
### Run the Code

- You can download the expert demonstrations used in our experiments from [this link](https://osf.io/fjvyw/) or generate new demonstrations through `metaworld_generate_expert/generate_demo.py`. Then place the `expert_demos` folder in `${root_dir}/IL`.
- Run experiments by the following command:
  ```
  python train.py agent=ot suite=metaworld obs_type=pixels suite/metaworld_task=hammer num_demos=10 seed=1 suite.num_train_frames=2000000 adaptive_discount=true
  ```
  The hyperparameter `adaptive_discount` controls whether to use Automatic Discount Scheduling.

## Citation
Please use the following bibtex for citations:

```
@article{liu2023imitation,
  title={Imitation Learning from Observation with Automatic Discount Scheduling},
  author={Liu, Yuyang and Dong, Weijun and Hu, Yingdong and Wen, Chuan and Yin, Zhao-Heng and Zhang, Chongjie and Gao, Yang},
  journal={arXiv preprint arXiv:2310.07433},
  year={2023}
}
```

## Acknowlegment
This codebase is built upon the [ROT](https://github.com/siddhanthaldar/ROT) codebase. The test environments are from [Meta-World](https://github.com/Farama-Foundation/Metaworld).