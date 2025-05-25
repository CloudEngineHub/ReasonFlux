<div align="center">
  <h1>ReasonFlux: Effective and Efficient Reasoning LLM Augmented with Thought Template</h1>
  <p>Revolutionary template-augmented reasoning paradigm enpowers a 32B model to achieve SOTA-Level performance in reasoning tasks.
 </p>
</div>


<p align="center">
<img src="./figs/comparison.png" width=80%>
</p>



## Table of Contents

- [Updates](#updates)
- [ReasonFlux-v2](./ReasonFlux_v2/README.md)
- [ReasonFlux-F1](./ReasonFlux_F1/README.md)
- [ReasonFlux-v1](./ReasonFlux_v1/README.md)
- [Performance](#performance)
- [Citation](#citation)

## Updates

- [2025/5/26] 🎉 We open-source the model weights of **Template-Proposer** and **Template-Reasoner**, training & evaluation scripts for ReasonFlux-v2. **We will release our paper related with ReasonFlux-V2 soon.**
- [2025/5/26] 🎉We release **ReasonFlux-v2**, an effective template-augmented reasoning paradigm that internalizes thought template through iterative hierarchical reinforcement learning. It has achieved SOTA-Level performance with less token consumption.
- [2025/3/24] 🎉We release [ReasonFlux-F1-32B](https://huggingface.co/Gen-Verse/ReasonFlux-F1), [ReasonFlux-F1-14B](https://huggingface.co/Gen-Verse/ReasonFlux-F1-14B), [ReasonFlux-F1-7B](https://huggingface.co/Gen-Verse/ReasonFlux-F1-7B), a series of SOTA-level reasoning LLMs by leveraging the template-augmented reasoning trajectories collected from our ReasonFlux-Zero. For the training and evaluation scripts, please refer to [reasonflux-f1/README.md](./reasonflux-f1/README.md) for detail.
- [2025/2/11] 🎉We release the data, training scripts for SFT stage and demo inference code along with template library of ReasonFlux-v1.
- [2025/2/11]🎉We propose [ReasonFlux-v1](), a hierarchical LLM reasoning framework that significantly enhances complex reasoning capabilities, outperforming SOTA models like o1-preview and DeepSeek-V3 on challenging MATH and AIME benchmarks.

## Performance

Her we compare our ReasonFlux series models with **Frontier LLMs** and other **Open-Sourced Reasoning LLMs** on challenging benchmarks like MATH-500,AIME2024,AIME-2025 and GPQA-Diamond. We can see that our method has achieved state-of-the-art performance on all evaluated tasks.

| Model                           | MATH-500 | AIME 2024 | AIME 2025 | GQPA-Diamond |
| ------------------------------- | -------- | --------- | --------- | ------------ |
| **Frontier LLMs**               |          |           |           |              |
| OpenAI-o1-2024-12-17            | 94.8     | 74.3      | 79.2      | –            |
| OpenAI-o3-mini (medium)         | 96.8     | 79.6      | 74.8      | 76.8         |
| Grok3 Beta                      | 96.6     | 83.9      | 77.3      | –            |
| Gemini 2.5-Pro                  | 98.4     | 92.0      | 86.7      | 84.0         |
| **Open-Sourced Reasoning LLMs** |          |           |           |              |
| DeepSeek-R1-Distill-7B          | 83.3     | 55.5      | 23.3      | 49.1         |
| DeepSeek-R1-Distill-14B         | 93.9     | 69.7      | 26.7      | 59.1         |
| DeepSeek-R1-Distill-32B         | 94.3     | 72.6      | 53.3      | 62.1         |
| DeepSeek-R1-Distill-70B         | 94.5     | 70.0      | 56.7      | 65.2         |
| DeepSeek-R1-67B                 | 97.3     | 79.8      | 70.0      | 71.5         |
| QwQ-32B-Preview                 | 90.6     | 50.0      | 46.7      | 65.2         |
| QwQ-32B                         | 97.6     | 80.0      | 63.3      | 68.18        |
| Qwen3-32B                       | 96.6     | 81.4      | 72.9      | 69.19        |
| Qwen3-30B-A3B                   | 96.8     | 80.4      | 70.9      | 65.8         |
| Qwen3-235B-A22B                 | 97.6     | 85.7      | **81.5**  | –            |
| Sky-T1-32B                      | 86.4     | 43.3      | 36.7      | 56.8         |
| LIMO-32B                        | 56.67    | 33.3      | 92.2      | 58.8         |
| s1.1-32B                        | 93.1     | 60.0      | 60.0      | 63.1         |
| OpenThinker-32B                 | 94.8     | 63.3      | 46.67     | 60.1         |
| Light-R1-32B                    | 96.2     | 78.1      | 68.0      | 60.1         |
| **ReasonFlux-V1 (2025-1)**      | **91.2** | **56.7**  | **37.2**  | **61.2**     |
| **ReasonFlux-F1 (2025-3）**     | **96.0** | **76.7**  | **53.3**  | **67.2**     |
| **ReasonFlux-V2 (2025-5)**      | **97.8** | **86.7**  | **76.7**  | **71.2**     |

## Citation

```bash
@article{yang2025reasonflux_v1,
  title={ReasonFlux: Hierarchical LLM Reasoning via Scaling Thought Templates},
  author={Yang, Ling and Yu, Zhaochen and Cui, Bin and Wang, Mengdi},
  journal={arXiv preprint arXiv:2502.06772},
  year={2025}
}
```
