# Training

We utilize open-source framework  [LLaMA-Factory]() to conduct our training process.

Step 1: Please add the data path to the file_name field of ReasonFlux entry in [LLaMA-Factory/data/dataset_info.json](./LLaMA-Factory/data/dataset_info.json).

Step 2: Run command below  to train from a 32B model on 8 A100 GPUs. 