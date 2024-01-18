[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/guess-gradually-enriching-synthesis-for-text/motion-synthesis-on-humanml3d)](https://paperswithcode.com/sota/motion-synthesis-on-humanml3d?p=guess-gradually-enriching-synthesis-for-text)

### [GUESS GradUally Enriching SyntheSis for Text-Driven Human Motion Generation](httpsarxiv.orgpdf2401.02142.pdf)  IEEE TVCG🔥

 GradUally Enriching SyntheSis (GUESS) is a powerful progressive text-to-motion genration strategy. It develops a cascaded latent diffusion model to improve motion realism and diversity.



## ⚡ Quick Start

### 1. Conda environment

```
conda create python=3.9 --name GUESS
conda activate GUESS
```

Install the packages in `requirements.txt` and install [PyTorch 1.12.1](httpspytorch.org)

```
pip install -r requirements.txt
```

We test our code on Python 3.9.12 and PyTorch 1.12.1.

### 2. Dependencies

Run the script to download dependencies materials

```
bash preparedownload_smpl_model.sh
bash prepareprepare_clip.sh
```

For Text to Motion Evaluation

```
bash preparedownload_t2m_evaluators.sh
```

### 3. Datasets
For convenience, you can download the datasets we processed directly. Please cite their oroginal papers if you use these datasets.


Datasets  Google Cloud                                        
-----------------------------------------------------------------------------------------------------------
 HumanML3D  [Download](httpsdrive.google.comdrivefolders1jjwwtyv6_rZzY7Bz60dEpOKIK9Fwh95Susp=drive_link) 
 KIT  [Download](httpsdrive.google.comdrivefolders1dh7zcwDz2M4yaE1Q9LWCHzghG-PWAkO4usp=drive_link) 

## 💻 Train your own models

details
  summarybWIPbsummary


### 1. Tran a VAE model for each skeleton scale

Please first check the parameters in `configsconfig_vae_humanml3d.yaml`, e.g. `NAME`,`DEBUG`.

Then, run the following command

```
python -m train --cfg configsconfig_vae_humanml3d.yaml --cfg_assets configsassets.yaml --batch_size 64 --nodebug
```

### 2. Train a cascaded diffusion model among scales

Please update the parameters in `configsconfig_mld_humanml3d.yaml`, e.g. `NAME`,`DEBUG`,`PRETRAINED_VAE` (change to your `latest ckpt model path` in previous step)

Then, run the following command

```
python -m train --cfg configsconfig_mld_humanml3d.yaml --cfg_assets configsassets.yaml --batch_size 64 --nodebug
```

### 3. Evaluate the model

Please first put the tained model checkpoint path to `TEST.CHECKPOINT` in `configsconfig_mld_humanml3d.yaml`.

Then, run the following command

```
python -m test --cfg configsconfig_mld_humanml3d.yaml --cfg_assets configsassets.yaml
```

details


## ▶️ Demo

details
  summarybText-to-motionbsummary

We support text file or keyboard input, the generated motions are npy files.
Please check the `configsasset.yaml` for path config, TEST.FOLDER as output folder.

Then, run the following script

```
python demo.py --cfg .configsconfig_mld_humanml3d.yaml --cfg_assets .configsassets.yaml --example .demoexample.txt
```

Some parameters

- `--example=.demoexample.txt` input file as text prompts
- `--task=text_motion` generate from the test set of dataset
- `--task=random_sampling` random motion sampling from noise
- ` --replication` generate motions for same input texts multiple times
- `--allinone` store all generated motions in a single npy file with the shape of `[num_samples, num_ replication, num_frames, num_joints, xyz]`

The outputs

- `npy file` the generated motions with the shape of (nframe, 22, 3)
- `text file` the input text prompt
details


## 👀 Visualization

details
  summarybDetails for Renderingbsummary

### 1. Setup blender - WIP

Refer to [TEMOS-Rendering motions](httpsgithub.comMathuxTEMOS) for blender setup, then install the following dependencies.

```
YOUR_BLENDER_PYTHON_PATHpython -m pip install -r preparerequirements_render.txt
```

### 2. (Optional) Render rigged cylinders

Run the following command using blender

```
YOUR_BLENDER_PATHblender --background --python render.py -- --cfg=.configsrender.yaml --dir=YOUR_NPY_FOLDER --mode=video --joint_type=HumanML3D
```

### 2. Create SMPL meshes with

```
python -m fit --dir YOUR_NPY_FOLDER --save_folder TEMP_PLY_FOLDER --cuda
```

This outputs

- `mesh npy file` the generate SMPL vertices with the shape of (nframe, 6893, 3)
- `ply files` the ply mesh file for blender or meshlab

### 3. Render SMPL meshes

Run the following command to render SMPL using blender

```
YOUR_BLENDER_PATHblender --background --python render.py -- --cfg=.configsrender.yaml --dir=YOUR_NPY_FOLDER --mode=video --joint_type=HumanML3D
```

optional parameters

- `--mode=video` render mp4 video
- `--mode=sequence` render the whole motion in a png image.
details


## 📌 Citation

If you find our code or paper helps, please consider citing

```
@ARTICLE{10399852,
  author={Gao, Xuehao and Yang, Yang and Xie, Zhenyu and Du, Shaoyi and Sun, Zhongqian and Wu, Yang},
  journal={IEEE Transactions on Visualization and Computer Graphics}, 
  title={GUESS GradUally Enriching SyntheSis for Text-Driven Human Motion Generation}, 
  year={2024}}
```

## Acknowledgments

Thanks to [MLD](httpsgithub.comChenFengYemotion-latent-diffusion), our code is partially borrowing from them.

## License

This code is distributed under an [MIT LICENSE](LICENSE).

Note that our code depends on other libraries, including SMPL, SMPL-X, PyTorch3D, and uses datasets which each have their own respective licenses that must also be followed.
