# TASER
## The code of "TASER: Efficient Black-Box Attack on Speech Platforms via Perturbing Linguistic Embedding".

Here we provide the code for generating the attack example described in the paper, and audio demos can be found on the [Demo Page](https://taser2023.github.io/).


## Descriptions of the code

- __taser_otl.py__: The implementation of __TASER-OTL__ which is described in the paper.

- __taser_otl_run.py__: Run __taser_otl.py__ and generate the example.

- __taser_ota.py__: The implementation of __TASER-OTA__ which is described in the paper.

- __taser_ota_run.py__: Run __taser_ota.py__ and generate the example.

- __optimizer.py__: We implement the particle swarm optimization here and use it in the __TASER-OTA__.

- __ASRs.py__: The speech-to-text APIs need to be implemented here.

- __tacotron2__: The repo of tacotron2 and we add some code at the bottom of __model.py__.

## Environment
- Python==3.6.13

- torch==1.8.0+cu111
  
You are supposed to build the tacotron2 environment by yourself and more details can be found in [Tacotron2](https://github.com/NVIDIA/tacotron2).
  
## How to generate the example

1. You need to implement the functions in __ASRs.py__ by yourself.

2. Set the parameters in __taser_otl_run.py__ or __taser_ota_run.py__.
   
3. Just run __taser_otl_run.py__ or __taser_ota_run.py__ and the results will be saved in the directory __taser_otl_example__ or __taser_ota_example__.

    `python taser_otl_run.py` or `python taser_ota_run.py`
