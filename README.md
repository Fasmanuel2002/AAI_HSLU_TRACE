### A PyTorch re-implementation of the [TRACE](https://arxiv.org/abs/2409.12972) paper, developed in collaboration with the university.
 <p align="center">
 <img width="700" height="627" alt="image" src="https://github.com/user-attachments/assets/340938a8-5ae1-44e3-824c-cf1d6e68f679" />
 </p>

This research project re-implements the **Transformer-based User Representation from Attributed Clickstream Event Sequence (TRACE)** architecture using alternative datasets, as the original dataset provided in the paper is not publicly available. TRACE proposes a novel approach for learning rich user representations from live, multi-session clickstream data with sparse targets.

The TRACE architecture is built around a multi-task learning (MTL) framework, in which a shared Transformer encoder is trained to predict multiple user engagement objectives from sequences of clickstream events. By predicting a diverse set of future user engagement signals, the architecture is encouraged to learn robust and versatile user representations that generalize across tasks and sessions.

<p align="center">
  <img width="682" alt="Training Pipeline" src="https://github.com/user-attachments/assets/4c82b31d-24c1-457d-8569-6bf3ca004317" />
</p>

The first step to run this project locally or on GPUHub is to install the required dependencies.

## Installation
```bash
pip install -r requirements.txt
