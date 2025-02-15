<div align="center">
  <h1>ReasonFlux: Hierarchical LLM Reasoning via Scaling Thought Templates</h1>
  <p>Revolutionary inference-scaling paradigm with a hierarchical RL algorithm: enpowering a 32B model with 500 thought templates to outperform o1-preview and DeepSeek-V3 in reasoning tasks.
 </p>
</div>

| Task           | **ReasonFlux 32B** | **DeepSeek V3** | **OpenAI o1-preview** | **OpenAI o1-mini** | **QWQ 32B-preview** | **s1-32B** |
| :------------- | :----------------: | :-------------: | :-------------------: | :----------------: | :-----------------: | :--------: |
| MATH500           |      **91.2**      |      90.2       |         85.5          |        90.0        |        90.6         |    84.8    |
| AIME 2024      |      **56.7**      |      39.2       |         44.6          |        56.7        |        50.0         |    36.0    |
| Olympiad Bench |      **63.3**      |      55.4       |           -           |        65.3        |        61.2         |    -    |
| GaokaoEn 2023  |      **83.6**      |        -        |         71.4          |        78.4        |        65.3         |    -    |
| AMC2023        |      **85.0**      |      80.0       |         90.0          |        95.0        |          -          |    -    |


<p align="center">
<img src="./figs/image.png" width=80%>
</p>

This repository provides resources for the paper ["ReasonFlux: Hierarchical LLM Reasoning via Scaling Thought Templates"](https://arxiv.org/abs/2502.06772).

- [Updates](#updates)
- [Dataset Links](#dataset-links)
- [Getting Started](#getting-started)
  - [Training](#training)
  - [Inference](#inference)
- [Performance](#performance)
- [Reasoning Example](#reasoning-example)
- [Preliminary Work](#preliminary-work)
- [Citation](#citation)

## Updates

- [2025/2~2025/5] We currently utilize our ReasonFlux to enhance reasoning capabilities of DeepSeek-R1, stay tuned for more updates.
- [2025/2/11] We release the data, training scripts for SFT stage and demo inference code along with template library of ReasonFlux.

## Dataset Links

- **[SFT Data of ReasonFlux](https://huggingface.co/datasets/Gen-Verse/ReasonFlux_SFT_15k)**

## Getting Started

```bash
git clone https://github.com/ReasonFlux
cd ReasonFlux
conda create -n ReasonFlux python==3.9
conda activate ReasonFlux
pip install -r requirements.txt
```

### Training

We utilize open-source framework  [LLaMA-Factory]() to conduct our training process.

Step 1: Please add the data path to the file_name field of ReasonFlux entry in [LLaMA-Factory/data/dataset_info.json](./LLaMA-Factory/data/dataset_info.json).

Step 2: Run command below  to train from a 32B model on 8 A100 GPUs. 

```bash
llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path Qwen/Qwen2.5-32B-Instruct \
    --preprocessing_num_workers 16 \
    --finetuning_type full \
    --template qwen \
    --flash_attn auto \
    --dataset_dir train/LLaMA-Factory/data \
    --dataset ReasonFlux \
    --cutoff_len 2048 \
    --learning_rate 2e-05 \
    --num_train_epochs 3.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing False \
    --report_to none \
    --output_dir saves/Qwen2.5-32B-Instruct/full \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --optim adamw_torch \
    --deepspeed cache/ds_z3_offload_config.json
```

### Inference

When you complete your first-stage training, you can try to use simple lines of codes to conduct reasoning based on few lines of code.

```python
from reasonflux import ReasonFlux

reasonflux = ReasonFlux(navigator_path='path-to-navigator',
                        template_matcher_path='jinaai/jina-embeddings-v3',
                 		inference_path='path-to-infernece-model',
                        template_path='template_library.json')
problem = """Given a sequence {aₙ} satisfying a₁=3, and aₙ₊₁=2aₙ+5 (n≥1), find the general term formula aₙ"""
```

`navigator_path` is the path to the navigator, you can put the path to your trained LLM after SFT-stage here.

`template_matcher_path` is the path to the embedding model, you can set the path to your local embedding model or download [jina-embedding-v3](https://huggingface.co/jinaai/jina-embeddings-v3) from [huggingface](https://huggingface.co/). 

`inference_path` is the path to the reasoning model, you can choose different-sized LLMs  to test but here we recommend you to choose the same LLMs as the navigator to save memory.

`template_path` is the path to our template library.  When you run the code for the first time, we will encode the template library for efficient query and retrieve and save the embedding in cache, and it is normal the first run will consume longer time in the initialization stage before reasoning.

You can test your trained model after the SFT stage to see if it could retrieve accurate templates given the problem and solve it in our demo implementation.

>🚨 It should be noted that if you choose to use jina-embedding-v3, you have to make sure that you do not install flash-attn in your environment, which will cause conflicts and thus fail to encode the query and the template library. 

## Performance

To evaluate the complex reasoning capabilities, we choose a broad set of challenging reasoning benchmarks, including MATH, AIME 2024, AMC 2023, OlympiadBench, and GaoKao (Chinese College Entrance Exam) En 2023. These benchmarks comprehensively evaluate mathematical reasoning capabilities, and they are all competition-level and Olympic-level problems. 

| Model                           | MATH     | AIME 2024 | AMC 2023                | Olympiad Bench | Gaokao En 2023 |
| :------------------------------ | :------- | :-------- | :---------------------- | :------------- | :------------- |
| **Frontier LLMs**               |          |           |                         |                |                |
| GPT-4o                          | 76.6     | 9.3       | 47.5                    | 43.3           | 67.5           |
| Claude3.5-Sonnet                | 78.3     | 16.0      | -                       | -              | -              |
| GPT-o1-preview                  | 85.5     | 44.6      | 90.0                    | -              | 71.4           |
| GPT-o1-mini                     | 90.0     | 56.7      | 95.0                    | 65.3           | 78.4           |
| **Open-Sourced Reasoning LLMs** |          |           |                         |                |                |
| DeepSeek-Coder-V2-Instruct      | 75.3     | 13.3      | 57.5                    | 37.6           | 64.7           |
| Mathstral-7B-v0.1               | 57.8     | 0.0       | 37.5                    | 21.5           | 46.0           |
| NuminaMath-72B-CoT              | 64.0     | 3.3       | 70.0                    | 32.6           | 58.4           |
| LLaMA3.1-8B-Instruct            | 51.4     | 6.7       | 25.0                    | 15.4           | 38.4           |
| LLaMA3.1-70B-Instruct           | 65.4     | 23.3      | 50.0                    | 27.7           | 54.0           |
| LLaMA3.1-405B-Instruct          | 73.8     | -         | -                       | 34.8           | -              |
| Qwen2.5-Math-72B-Instruct       | 85.6     | 30.0      | 70.0                    | 49.0           | 71.9           |
| rStar-Math                      | 88.2     | 43.3      | 80.0                    | 63.1           | 78.2           |
| DeepSeek-V3                     | 90.2     | 39.2      | 80.0                    | 55.4           | -              |
| **ReasonFlux-32B**              | **91.2** | **56.7**  | **85.0**                | **63.3**       | **83.6**       |
|                                 |          |           | *1.5B-Level Base Model* |                |                |
| Qwen2.5-Math-1.5B               | 51.2     | 0.0       | 22.5                    | 16.7           | 46.5           |
| Qwen2.5-Math-1.5B-Instruct      | 60.0     | 10.0      | 60.0                    | 38.1           | 65.5           |
| **ReasonFlux-1.5B**             | **70.4** | **20.0**  | **72.5**                | **49.0**       | **76.6**       |
|                                 |          |           | *7B-Level Base Model*   |                |                |
| Qwen2.5-Math-7B                 | 58.8     | 3.3       | 22.5                    | 21.8           | 51.7           |
| SuperCorrect-7B                 | 70.2     | 10.0      | 37.5                    | 39.0           | 64.0           |
| Qwen2.5-Math-7B-Instruct        | 82.6     | 13.3      | 62.5                    | 41.6           | 66.8           |
| **ReasonFlux-7B**               | **88.6** | **36.7**  | **80.0**                | **54.8**       | **80.5**       |
|                                 |          |           | *32B-Level Base Model*  |                |                |
| Qwen2.5-32B-Instruct            | 79.4     | 16.5      | 64.0                    | 45.3           | 72.1           |
| QwQ-32B-preview                 | 90.6     | 50.0      | 75.0                    | -              | 65.3           |
| Sky-T1-32B-preview              | 86.4     | 43.3      | -                       | 59.8           | -              |
| **ReasonFlux-32B**              | **91.2** | **56.7**  | **85.0**                | **63.3**       | **83.6**       |

## Reasoning Example

![example](figs/example.png)

## Preliminary Work
We build our ReasonFlux mainly based on some preliminary works, such as [Buffer of Thoughts](https://github.com/YangLing0818/buffer-of-thought-llm) and [SuperCorrect](https://github.com/YangLing0818/SuperCorrect-llm).

## Citation

```bash
@article{yang2025reasonflux,
  title={ReasonFlux: Hierarchical LLM Reasoning via Scaling Thought Templates},
  author={Yang, Ling and Yu, Zhaochen and Cui, Bin and Wang, Mengdi},
  journal={arXiv preprint arXiv:2502.06772},
  year={2025}
}
```
