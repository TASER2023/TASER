# ALIF
## The code of "ALIF: Low-Cost Adversarial Audio Attacks on Black-Box Speech Platforms Using Linguistic Features".

Here we provide the code for generating the attack examples described in the paper, and audio demos can be found on the [Demo Page](https://taser2023.github.io/).


## Descriptions of the code

- __alif_otl.py__: The implementation of __ALIF-OTL__ which is described in the paper.

- __alif_otl_run.py__: Run __alif_otl.py__ and generate the example.

- __alif_ota.py__: The implementation of __ALIF-OTA__ which is described in the paper.

- __alif_ota_run.py__: Run __alif_ota.py__ and generate the example.

- __optimizer.py__: We implement the particle swarm optimization here and use it in the __ALIF-OTA__.

- __ASRs.py__: The speech-to-text APIs need to be implemented here.

- __tacotron2__: The repo of tacotron2 and we add some code at the bottom of __model.py__.

## Environment
- Python==3.6.13

- torch==1.8.0+cu111
  
You are supposed to build the tacotron2 environment by yourself and more details can be found in [Tacotron2](https://github.com/NVIDIA/tacotron2).
  
## How to generate the example

1. You need to implement the functions in __ASRs.py__ by yourself.

2. Set the parameters in __alif_otl_run.py__ or __alif_ota_run.py__.
   
3. Just run __alif_otl_run.py__ or __alif_ota_run.py__ and the results will be saved in the directory __alif_otl_example__ or __alif_ota_example__.

    `python alif_otl_run.py` or `python alif_ota_run.py`
