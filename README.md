# 🧠 ReasonFlux Series
### *Advanced Open-Source LLM Post-Training Suite*
**Princeton University** \& **PKU** \& **UIUC** \& **University of Chicago** \& **ByteDance Seed**

**🎯 Mission**: Building next-generation reasoning capabilities through innovative LLM post-training algorithms focusing on **data selection**, **reinforcement learning**, and **inference scaling**.

## Contents of Repository

- [Updates](#updates)
- [Model Family Guide](#model-family-guide)
  - [ReasonFlux-PRM **(NeurIPS 2025)**](./ReasonFlux_PRM/README.md)
  - [ReasonFlux-Coder **(NeurIPS 2025 Spotlight)**](https://github.com/Gen-Verse/CURE)
  - [ReasonFlux](./ReasonFlux/README.md)
  - [Preliminary Work on Thought Template **(NeurIPS 2024 Spotlight)**](#preliminary-work-on-thought-template)
- [Performance Hightlights](#performance-highlights)
- [Citation](#citation)

## 🚀 What Makes ReasonFlux Series Special?

### 1. Trajectory-Aware Process Reward Models for Long-CoT Reasoning (ReasonFlux-PRM, NeurIPS 2025)
Trajectory-aware reward models that provide dense supervision for both offline data selection and online policy optimization in long-CoT reasoning.
<p align="center">
<img src="./ReasonFlux_PRM/img/intro_res.png" width=100%>
</p>

### 2. Co-Evolved RL for LLM Coder and Unit Tester (ReasonFlux-Coder, NeurIPS 2025 Spotlight)
Innovative approach where coders and unit testers evolve together through reinforcement learning, creating more robust coding capabilities.
<p align="center">
<img src="./ReasonFlux_Coder/figures/overviewplot.png" width=100%>
</p>

### 3. Long-CoT Reasoning with Thought Templates (ReasonFlux-Zero/F1)
Revolutionary hierarchical reasoning framework that uses thought templates to guide complex problem-solving, achieving SOTA performance with higher efficiency.

<p align="center">
<img src="./figs/comparison.png" width=100%>
</p>


## Preliminary Work on Thought Template
Our ReasonFlux-Zero/F1 models are built upon insights from our preliminary work on thought templates—specifically, [Buffer of Thoughts (NeurIPS 2024 Spotlight)](https://openreview.net/forum?id=ANO1i9JPtb) and [SuperCorrect (ICLR 2025)](https://openreview.net/forum?id=PyjZO7oSw2). These works introduce high-level, efficient intermediate reasoning patterns that guide and structure the thinking process of large language models.


## Updates

- [2025/6/23] 🎉 We introduce [**ReasonFlux-PRM**](https://arxiv.org/abs/2506.18896), a family of trajectory-aware process reward models (PRMs) for long CoT reasoning in LLMs. ReasonFlux-PRM is able to support **both offline and online reward supervision**, by selecting high-quality training data for model distillation, providing dense process-level rewards for policy optimization during reinforcement learning, and enabling reward-guided test-time scaling. 
Our trained PRMs including [ReasonFlux-PRM-7B](https://huggingface.co/Gen-Verse/ReasonFlux-PRM-7B) and [ReasonFlux-PRM-1.5B](https://huggingface.co/Gen-Verse/ReasonFlux-PRM-1.5B) are now available on [HuggingFace-GenX](https://huggingface.co/Gen-Verse). We also release a 7B advanced thinking and reasoning model [ReasonFlux-PRM-Qwen-2.5-7B](https://huggingface.co/Gen-Verse/ReasonFlux-PRM-Qwen-2.5-7B) supervised via our PRM.
- [2025/6/04] 🎉 We release our [**Co-Evolving RL**](https://github.com/Gen-Verse/CURE) optimized coding LLMs, [ReasonFlux-Coder-7B](https://huggingface.co/Gen-Verse/ReasonFlux-Coder-7B) and [ReasonFlux-Coder-14B](https://huggingface.co/Gen-Verse/ReasonFlux-Coder-14B), which outperform similarly sized Qwen Coders and DeepSeek Coders, and naturally fit into common test-time scaling and agentic coding pipelines. We also release our Long-CoT model [ReasonFlux-Coder-4B](https://huggingface.co/Gen-Verse/ReasonFlux-Coder-4B), outperforming Qwen3-4B while achieving 64.8% efficiency in unit test generation.
- [2025/3/24] 🎉We release [ReasonFlux-F1-32B](https://huggingface.co/Gen-Verse/ReasonFlux-F1), [ReasonFlux-F1-14B](https://huggingface.co/Gen-Verse/ReasonFlux-F1-14B), [ReasonFlux-F1-7B](https://huggingface.co/Gen-Verse/ReasonFlux-F1-7B), a series of SOTA-level reasoning LLMs by leveraging the template-augmented reasoning trajectories collected from our ReasonFlux-Zero. For the training and evaluation scripts, please refer to [reasonflux-f1/README.md](./ReasonFlux/README.md) for detail.
- [2025/2/11]🎉We propose [ReasonFlux-Zero](https://arxiv.org/abs/2502.06772), a hierarchical LLM reasoning framework that significantly enhances complex reasoning capabilities, outperforming SOTA models like o1-preview and DeepSeek-V3 on challenging MATH and AIME benchmarks.

## Model Family Guide



### 🎯 **Process Reward Models**

<table>
<tr>
<th>Model</th>
<th>Size</th>
<th>Capabilities</th>
<th>Use Cases</th>
<th>Download</th>
</tr>
<tr>
<td><strong>ReasonFlux-PRM</strong></td>
<td>7B</td>
<td>• Trajectory-aware scoring<br/>• Online/Offline supervision<br/>• Dense process rewards</td>
<td>PRM: Data selection, RL training, Test-time scaling</td>
<td><a href="https://huggingface.co/Gen-Verse/ReasonFlux-PRM-7B">🤗 7B</a></td>
</tr>
<tr>
<td><strong>ReasonFlux-PRM</strong></td>
<td>1.5B</td>
<td>• Lightweight scoring<br/>• Efficient inference<br/>• Edge deployment</td>
<td>PRM: Resource-constrained applications</td>
<td><a href="https://huggingface.co/Gen-Verse/ReasonFlux-PRM-1.5B">🤗 1.5B</a></td>
</tr>
</tr>
<tr>
<td><strong>ReasonFlux-PRM-Qwen-2.5</strong></td>
<td>7B</td>
<td>• Long CoT reasoning <br/>• Solving complex tasks and problems</td>
<td>Tuned Reasoning Model: Math and Science Reasoning</td>
<td><a href="https://huggingface.co/Gen-Verse/ReasonFlux-PRM-Qwen-2.5-7B">🤗 7B</a></td>
</tr>
</table>

### 💻 **Coding Models**

<table>
<tr>
<th>Model</th>
<th>Size</th>
<th>Specialization</th>
<th>Performance</th>
<th>Download</th>
</tr>
<tr>
<td><strong>ReasonFlux-Coder</strong></td>
<td>14B</td>
<td>• Co-evolutionary RL<br/>• Advanced coding<br/>• Unit test generation</td>
<td>Outperforms Qwen & DeepSeek Coders</td>
<td><a href="https://huggingface.co/Gen-Verse/ReasonFlux-Coder-14B">🤗 14B</a></td>
</tr>
<tr>
<td><strong>ReasonFlux-Coder</strong></td>
<td>7B</td>
<td>• Balanced performance<br/>• Efficient inference<br/>• Production ready</td>
<td>Excellent coding capabilities</td>
<td><a href="https://huggingface.co/Gen-Verse/ReasonFlux-Coder-7B">🤗 7B</a></td>
</tr>
<tr>
<td><strong>ReasonFlux-Coder</strong></td>
<td>4B</td>
<td>• Long-CoT reasoning<br/>• Compact size<br/>• Unit test focused</td>
<td>64.8% efficiency in unit test generation</td>
<td><a href="https://huggingface.co/Gen-Verse/ReasonFlux-Coder-4B">🤗 4B</a></td>
</tr>
</table>


### 🧠 **Reasoning Models**

<table>
<tr>
<th>Model</th>
<th>Size</th>
<th>Key Features</th>
<th>Best For</th>
<th>Download</th>
</tr>
<tr>
<td><strong>ReasonFlux-F1</strong></td>
<td>7B/14B/32B</td>
<td>• Template-augmented trajectories<br/>• Efficient training<br/>• Multiple sizes</td>
<td>General reasoning tasks</td>
<td><a href="https://huggingface.co/collections/Gen-Verse/reasonflux-series-67e8ebd46c7216f5bf8c2421">🤗 Models</a></td>
</tr>
<tr>
<td><strong>ReasonFlux-Zero</strong></td>
<td>32B</td>
<td>• Hierarchical reasoning<br/>• Template library<br/>• Foundation model</td>
<td>Research & development</td>
<td><a href="#">🤗 Model</a></td>
</tr>
</table>


## Performance Highlights

### 1. Complex Reasoning

| Model                 | AIME2024@pass1 | AIME2025@pass1 | MATH500@pass1 | GPQA@pass1 |
| --------------------- | :------------: | :------------: | :-----------: | :--------: |
| QwQ-32B-Preview       |      46.7      |      37.2      |     90.6      |    65.2    |
| LIMO-32B              |      56.3      |      44.5      |     94.8      |    58.1    |
| s1-32B                |      56.7      |      49.3      |     93.0      |    59.6    |
| OpenThinker-32B       |      66.0      |      53.3      |     94.8      |    60.1    |
| R1-Distill-32B        |      70.0      |      46.7      |     92.0      |    59.6    |
| ReasonFlux-Zero-32B   |      56.7      |      37.2      |     91.2      |    61.2    |
| **ReasonFlux-F1-32B** |    **76.7**    |    **53.3**    |   **96.0**    |  **67.2**  |


### 2. Code Generation and Reasoning
<p align="center">
  <img src="ReasonFlux_Coder/figures/results.png"   alt="Results of ReasonFlux-Coder"  width="700">
</p>

### 3. PRMs for Long-CoT Reasoning
We observe that in the downstream offline data selection + SFT setting, ReasonFlux-PRM-7B surpasses the performance of the high-quality, human-curated s1k dataset. We further visualize the score distributions over 1,000 trajectory-response pairs generated by Deepseek-R1 and Gemini. The clearly separated distributions indicate that ReasonFlux-PRM-7B effectively differentiates the quality of responses from different models, offering a robust and reliable reward signal for high-quality data selection.

<img src="./ReasonFlux_PRM/img/sft.png" alt="" style="width: 100%; max-width: 1000px; margin-bottom: 20px;" id="sft">

Under the online settings, ReasonFlux-PRM-7B also surpasses other PRM and rule-based baselines during the GRPO policy optimization. 

<img src="./ReasonFlux_PRM/img/rl.png" alt="" style="width: 100%; max-width: 1000px; margin-bottom: 20px;" id="rl">



<img src="./ReasonFlux_PRM/img/efficiency.png" alt="" style="width: 100%; max-width: 1000px; margin-bottom: 10px;" id="efficiency">

## Citation

```bash
@article{yang2025reasonflux,
  title={ReasonFlux: Hierarchical LLM Reasoning via Scaling Thought Templates},
  author={Yang, Ling and Yu, Zhaochen and Cui, Bin and Wang, Mengdi},
  journal={arXiv preprint arXiv:2502.06772},
  year={2025}
}

@article{wang2025reasonfluxcoder,
  title={Co-Evolving LLM Coder and Unit Tester via Reinforcement Learning},
  author={Wang, Yinjie and Yang, Ling and Tian, Ye and Shen, Ke and Wang, Mengdi},
  journal={arXiv preprint arXiv:2506.03136},
  year={2025}
}

@article{zou2025reasonfluxprm,
  title={ReasonFlux-PRM: Trajectory-Aware PRMs for Long Chain-of-Thought Reasoning in LLMs},
  author={Zou, Jiaru and Yang, Ling and Gu, Jingwen and Qiu, Jiahao and Shen, Ke and He, Jingrui and Wang, Mengdi},
  journal={arXiv preprint arXiv:2506.18896},
  year={2025}
}
```

