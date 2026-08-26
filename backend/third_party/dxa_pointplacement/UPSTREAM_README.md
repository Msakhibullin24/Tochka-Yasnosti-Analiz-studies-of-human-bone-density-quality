# dxa-pointplacement
Repository for fiducial point placement on total body DXA scans. This code creates point files for air ratio DXA images (not tested on other extracted image types, but let us know how it goes if you try it on another extraction format) designed to be fed into the [UoMApM](https://uomapm.sourceforge.io/) tools for shape and appearance modeling. Points can be used on any registered image mode by simply pointing your images/ directory to the correct mode. You can also use this to place points without feeding into the PCA modeling pipeline provided by UoMApM, but the pts file format may need to be amended for your purposes. Model checkpoint files (for the results reported in the paper) can be downloaded from [here](https://drive.google.com/drive/folders/1gbXTJfU3RGYNr1lvEhDJA1qlXRb3NwYW?usp=drive_link).

###### Github repository containing all relevant code for ShapeMI at MICCAI 2025 submission

## Installation and system requirements
- Tested on Ubuntu 20.04.6 LTS
- Python version: 3.8.16
- Install dependencies
  * `opencv-python==4.7.0.72`
  * `numpy==1.24.3`
  * `pillow==9.4.0`
  * `albumentations==1.3.1` (if training)
- Install mmengine and mmcv as per the instructions on the [mmpose website](https://mmpose.readthedocs.io/en/latest/installation.html), copied below for convenience. Does not need to be installed with GPU support if you are not developing/finetuning for your own use.
```
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.1"
```
- Install mmpose. If you are planning on finetuning this model for your own use, I would recommend installing from source as per the instructions on the [mmpose website](https://mmpose.readthedocs.io/en/latest/installation.html), copied below for convenience. Otherwise, you can install using conda or pip. Does not need to be installed with GPU support if you are not developing/finetuning for your own use. If you are developing/finetuning, dxa_points.py needs to be moved to datasets/datasets in the mmpose source code structure and dxa_points_config.py needs to be moved to configs/\_base\_.
```
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -r requirements.txt
pip install -v -e .
```
- (if using UoMApM tools) Install UoMApM using the instructions [here](https://uomapm.sourceforge.io/installing_uomapm.html). These instructions were last tested in August 2024.
## Demo
* A demo inference script is provided in the outermost folder. This script runs the sample scans through a pretrained model and creates pts files for them. **These are the same pts files as in the points directory.** You can check your model functionality by verifying these values are the same. The bottom section of this file walks through how to then move this output to be used in UoMApM tools.
   * Run the inference file as below
    ```
    python dxa_inference.py demo [gpu index] [visualization 1/0]
    ```
    * `dxa_inference.py` relies on the function `define_paths` to define what metadata file to use, where your images are, where to save pts files, and if your images need site-specific naming (critically important if images come from different centers). Please modify this as necessary for your own uses. The demo dataset can be called with 'demo' as the first command-line argument.
    * `dxa_points_config.py` and `dxa_points.py1 are files for openmmlab to define the structure of the DXA files and the pretrained model. If you are going to be training your own model, I would suggest reviewing these to make sure all configurations match your expectations.
* A demo dataset is provided purely to validate model functionality (images directory), the dataset is not representative of the complete dataset used to train/evaluate the models in the manuscript. All included sample scans are male and so demo scripts only use males.tri.
## Moving to UoMApM
The DXA point placement model only gets you halfway to doing the PCA analysis which is presented in the paper for this repository. The second step is to use the [UoMApM](https://uomapm.sourceforge.io/) tools for shape and appearance modeling. If you use the `dxa_inference.py` script with `sample_point_placement.csv` and our sample data, and this cloned repository, you'll end up with populated points/ images/ and models/ directories. The instructions below assume this structure, and that you have the UoMApM tools installed. All credit for UoMApM tools goes to Tim Cootes and team.
* Demo triangulation files are provided as males.tri and females.tri, demo app building and extraction configuration files are provided as build_app-tri.params and get-apm-params.params.
* To train your PCA model, run `mapm_build_app_model -p build_app-tri.params -o tri-grey.mapm` in your command line.
* To extract PCs from your trained PCA model and images, run `mapm_get_apm_params -p get_apm_params.params -i tri-grey.mapm -o males`
* To create mode images as presented in the paper, run `mapm_make_mode_images -apm tri-grey.mapm -t app -nm 3 -o males-app`
