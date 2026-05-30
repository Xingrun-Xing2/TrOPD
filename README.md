<h1 align="center">Trust Region On-Policy Distillation</h1>

<!--
<p align="center">
    <a href="https://arxiv.org/abs/2502.06663">
            <img alt="Build" src="https://img.shields.io/badge/arXiv%20paper-2502.06663-b31b1b.svg">
    </a>
    <a href="https://huggingface.co/collections/xrxing/efficientllm-pruning-aware-pretraining-67a8ecc6a49580b647a6184f">
        <img alt="Build" src="https://img.shields.io/badge/HF%20Model-🤗-yellow">
    </a>
</p>
-->

This repository contains the training code and models of TrOPD introduced in our work: ["Trust Region On-Policy Distillation"](https://arxiv.org/abs/xxx).

<!--
## News
- Feb 10, 2025: 🚀 100M ~ 1B edge models are publicly available on [HuggingFace](https://huggingface.co/collections/xrxing/efficientllm-pruning-aware-pretraining-67a8ecc6a49580b647a6184f).
-->

## 1. Overview

On-Policy Distillation (OPD) is a fundamental technique for efficient post-training of large language models (LLMs), with broad applications in agent learning, multi-task enhancement, and model compression. However, OPD training becomes unstable when the teacher and student distributions differ substantially, as teacher supervision on student-generated tokens may yield unreliable policy gradients and even cause optimization failure.
This work addresses reliable on-policy token-level supervision through credit assignment strategies, and proposes Trust Region On-Policy Distillation, TrOPD. 
It features the following characteristics:
\textbf{1) Trust-Region On-Policy Learning:} TrOPD performs OPD only in regions where the teacher provides reliable supervision, mitigating the optimization difficulty of the $K_1$ reverse-KL estimator under distribution mismatch. 
\textbf{2) Outlier Estimation:} For outlier regions, we explore gradient clipping, masking, and forward-KL estimation to reduce the adverse effects of unreliable supervision. 
\textbf{3) Off-Policy Guidance:} The student continues generation from teacher prefixes and uses forward KL to imitate off-policy guidance, encouraging on-policy exploration toward reliable regions.
Experiments show that TrOPD consistently outperforms SoTA OPD baselines, including OPD, EOPD, and REOPOLD, across mathematical reasoning, code generation, and general-domain benchmarks.

<div align=center>
<img width=90% src="https://github.com/Xingrun-Xing2/TrOPD/blob/main/imgs/f2.pdf"/>
</div>

**Figure 1**: 
Overview of Trust Region On-Policy Distillation. 
For the on-policy component, student-generated tokens are divided into the trust region and outliers.
The student model is further guided by teacher-generated responses.

## 2. Results of OPD Benchmarks

<div align=center>
<img width=98% src="https://github.com/Xingrun-Xing2/TrOPD/blob/main/imgs/f-1.pdf"/>
</div>


<!--
## 3. Auto-Designed Architecture

<div align=center>
<img width=98% src="https://github.com/Xingrun-Xing2/EfficientLLM/blob/main/imgs/figa0.png"/>
</div>

## 4. Load Huggingface Models

To load a pre-trained model and tokenizer, you can use the following code snippet:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("xrxing/EfficientLLM-469M", use_fast=False)

# Load the model
model = AutoModelForCausalLM.from_pretrained("xrxing/EfficientLLM-469M", trust_remote_code=True, attn_implementation="flash_attention_2")
```
-->

## 3. ToDo List

- [x] Release technical report
- [ ] Release Huggingface models
- [ ] Training and evaluation code
- [ ] Demos and applications


## Contact

Xingrun Xing, Samsung Research, Beijing, China (xingrun.xing@partner.samsung.com)

<!--
## Citation
If you find this work useful for your research, please consider citing:
```
@misc{xing2025efficientllm,
      title={EfficientLLM: Scalable Pruning-Aware Pretraining for Architecture-Agnostic Edge Language Models}, 
      author={Xingrun Xing and Zheng Liu and Shitao Xiao and Boyan Gao and Yiming Liang and Wanpeng Zhang and Haokun Lin and Guoqi Li and Jiajun Zhang},
      year={2025},
      eprint={2502.06663},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.06663}, 
}
```
-->

