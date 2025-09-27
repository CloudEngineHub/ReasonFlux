# Evaluate ReasonFlux on the AIME-24, AIME-25, OpenAI Math, and GPQA Diamond OpenAI tasks
OPENAI_API_KEY=sk-xxxx  lm_eval --model vllm --model_args pretrained=Gen-verse/ReasonFlux-F1,dtype=float32,tensor_parallel_size=8,gpu_memory_utilization=0.95 --tasks aime24_nofigures,aime25_nofigures,openai_math,gpqa_diamond_openai --batch_size auto --apply_chat_template --output_path ReasonFlux-F1 --log_samples --gen_kwargs "max_gen_toks=32768"

# Evaluate ReasonFlux on AIME-24 only
OPENAI_API_KEY=sk-xxxx  lm_eval --model vllm --model_args pretrained=Gen-verse/ReasonFlux-F1,dtype=float32,tensor_parallel_size=8,gpu_memory_utilization=0.95 --tasks aime24_nofigures --batch_size auto --apply_chat_template --output_path ReasonFlux-F1 --log_samples --gen_kwargs "max_gen_toks=32768"

# Evaluate ReasonFlux on AIME-25 only
OPENAI_API_KEY=sk-xxxx  lm_eval --model vllm --model_args pretrained=Gen-verse/ReasonFlux-F1,dtype=float32,tensor_parallel_size=8,gpu_memory_utilization=0.95 --tasks aime25_nofigures --batch_size auto --apply_chat_template --output_path ReasonFlux-F1 --log_samples --gen_kwargs "max_gen_toks=32768"

# Evaluate ReasonFlux on OpenAI Math only
OPENAI_API_KEY=sk-xxxx  lm_eval --model vllm --model_args pretrained=Gen-verse/ReasonFlux-F1,dtype=float32,tensor_parallel_size=8,gpu_memory_utilization=0.95 --tasks openai_math --batch_size auto --apply_chat_template --output_path ReasonFlux-F1 --log_samples --gen_kwargs "max_gen_toks=32768"

# Evaluate ReasonFlux on GPQA Diamond OpenAI only
OPENAI_API_KEY=sk-xxxx  lm_eval --model vllm --model_args pretrained=Gen-verse/ReasonFlux-F1,dtype=float32,tensor_parallel_size=8,gpu_memory_utilization=0.95 --tasks gpqa_diamond_openai --batch_size auto --apply_chat_template --output_path ReasonFlux-F1 --log_samples --gen_kwargs "max_gen_toks=32768"