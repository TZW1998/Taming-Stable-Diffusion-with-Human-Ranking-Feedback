# Taming Stable Diffusion with Human Ranking Feedback



This is the official repo for the paper "Zeroth-Order Optimization Meets Human Feedback: Provable Learning via  Ranking Oracles" Tang et al. https://arxiv.org/abs/2303.03751





## Intro

In this work, we invent a new zero order optimization algorithm that can optimize any function via only its ranking oracle. More importantly, we successfully apply our algorithm to a novel application shown in the following figure, where we optimize the latend embedding of Stable Diffusion with human ranking feedback. Specifically, starting from the latent embedding of an initial image, we first perturbed the embedding with multiple random noise vectors and then use Stable Diffusion to generate multiple simialr images (only differ in details). Then we ask some human evaluator (ourselves actually) to rank those generated image. Finally, our algorithm will update the latent embedding based on the ranking information. **Notice: our method does not require any training or finetuning at all!**. From our experience, it usually takes around 10-20 rounds of ranking feedback before obtaining image with satisfying details.

![overview](overview.png)



## Some examples

We provide some examples below, where the column under "Human" meaning the images obtained by optimizing human preference, while the column under "CLIP" meaning the ones by optimizing CLIP score. We use some popular prompts from this website https://mpost.io/best-100-stable-diffusion-prompts-the-most-beautiful-ai-text-to-image-prompts . More examples can be found in the paper

![example1](example1.png)





## Running our code

Our main code is provided as the ipynb script: stable_diffusion_alignment.ipynb



To run our code, you should first install Stable Diffusion properly following the guidance in https://github.com/CompVis/stable-diffusion .



We actually use the Oneflow version of Stable Diffusion in our implementation https://github.com/Oneflow-Inc/diffusers/tree/oneflow-fork, as we found that it can generate much faster than the official implementation.





## ToDo

- [ ] Intergate it into Stable Diffusion Webui

## Contact

Zhiwei Tang, zhiweitang1@link.cuhk.edu.cn



## Cite

If our work is useful for you, please cite our paper
```
@misc{https://doi.org/10.48550/arxiv.2303.03751,
doi = {10.48550/ARXIV.2303.03751},
url = {https://arxiv.org/abs/2303.03751},
author = {Tang, Zhiwei and Rybin, Dmitry and Chang, Tsung-Hui},
keywords = {Machine Learning (cs.LG), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
title = {Zeroth-Order Optimization Meets Human Feedback: Provable Learning via Ranking Oracles},
publisher = {arXiv},
year = {2023},
}

````
