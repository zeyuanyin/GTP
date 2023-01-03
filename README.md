# Generate Targeted Perturbations: Targeted Adversarial Attack via Generative Perturbations

> code is modified from [On Generating Transferable Targeted Perturbations (ICCV'21)](https://github.com/Muzammal-Naseer/TTP)

## Train Data Setting

You can run the following command on your own ImageNet data to create the train data (taking one hour to run):

```bash
python create_traindata.py --imagenet_path /path/to/your/imagenet/train --dest_dir /path/to/your/imagenet/train_50
```
Or you can download the processed train data [here](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/zeyuan_yin_mbzuai_ac_ae/Ec416z_rKoBMijojjDNGJ9cBT8wlBdCmuxWb-CU4NtF2zg?e=39YmQJ) (5.3G), which contain 50K images randomly selected from ImageNet train data (~1.2M).


## Train Generators

**data setting**

Using 49.95K images in /imagenet/train_50 as source data (train data), 999 source classes, 50 images per class

Using 1300 images in /imagenet/train as target data, 1 target class, 1300 images




## Evaluate Generators

**data setting**

Using 450 images in /imagenet/eval as source data (test data), 9 source classes, 50 images per class