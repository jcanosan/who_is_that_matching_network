# "Who is that Pokémon?" with Matching Networks

This project is under development.

This project provides an implementation of Matching Networks as described in the paper [Matching Networks for One Shot Learning](http://papers.nips.cc/paper/6385-matching-networks-for-one-shot-learning.pdf) in PyTorch.

This is based in the [implementation](https://github.com/jcanosan/Interactive-robot-with-neural-networks) made for the paper [Fast visual grounding in interaction: bringing few-shot learning with neural networks to an interactive robot](https://aclanthology.org/2020.pam-1.7/), which stems from my Master's Thesis project for the Master's in Language Technology at the University of Gothenburg.

## Usage
1. Install all the required Python (3.12) libraries in requirements.txt.

2. (As of now) Run "image_dataset.py"
```python image_dataset.py```

## SOTA dataset  # TODO eventually remove from this repo
SOTA (Small Objects daTAset) is a dataset of 400 images distributed equally into 20 categories. These images portrait a single object of interest which is normally centered in the image.

The images were taken using the same Kinect v1 RGB camera. The images taken are resized to 224x224 pixels since it is the default size that VGG16 and most deep image encoders use by default.

SOTA is available inside this same repository. 

This dataset is licensed under Creative Commons Attribution 4.0 International (CC BY 4.0). Anyone is free to share an adapt this dataset as long as appropriate credit is given to the original author.

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" href="http://purl.org/dc/dcmitype/StillImage" property="dct:title" rel="dct:type">Small Objects daTAset (SOTA)</span> by <a xmlns:cc="http://creativecommons.org/ns#" href="https://github.com/jcanosan/Interactive-robot-with-neural-networks/tree/master/utils/datasets/sota_dataset" property="cc:attributionName" rel="cc:attributionURL">José Miguel Cano Santín</a> is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
