# ReasonFlux-F1

## Training

We use fork from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to perform training.

To train ReasonFlux-F1, you should follow the steps above:

**Step 1:** Please add the data path to the file_name field of ReasonFlux-F1 entry in [LLaMA-Factory/data/dataset_info.json](./LLaMA-Factory/data/dataset_info.json).

**Step 2:** Run the command below to train ReasonFlux-F1-32B. 

```bash
llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --preprocessing_num_workers 16 \
    --finetuning_type full \
    --template qwen \
    --flash_attn auto \
    --dataset_dir data \
    --dataset ReasonFlux-F1 \
    --cutoff_len 16384 \
    --learning_rate 1e-05 \
    --num_train_epochs 5.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing False \
    --report_to none \
    --output_dir saves/DeepSeek-R1-Distill-Qwen-32B/full/ReasonFlux-F1 \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --deepspeed cache/ds_z3_offload_config.json
```

## Evaluation

We cloned [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) at commit `4cec66e4e468d15789473d6d63c3a61a751fa524` and modified it. Setup:
```bash
cd eval/lm-evaluation-harness
pip install -e .[math,vllm]
```

All commands are in `eval/commands.sh`. For AIME24 we always pick the `aime24_nofigures` result, which uses a dataset that only contains the AIME24 figures if they are important for the task.

For example, to evaluate ReasonFlux-F1-32B on AIME24/25, MATH500 and GPQA-Diamond, you can use the command below:

```bash
OPENAI_API_KEY=Input your openai key here lm_eval --model vllm --model_args pretrained=Gen-verse/ReasonFlux,dtype=float32,tensor_parallel_size=8,gpu_memory_utilization=0.95 --tasks aime24_figures,aime25_nofigures,openai_math,gpqa_diamond_openai --batch_size auto --apply_chat_template --output_path ReasonFlux-F1 --log_samples --gen_kwargs "max_gen_toks=32768"
```

 