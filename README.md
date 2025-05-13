# Training Code

## Usage

### Installation

- 8 x NVIDIA A800-80GB
- CUDA 12.1
- Python 3.10.16
- Torch 2.5.1+cu121

```bash
conda create -n finetune python=3.10 -y
conda activate finetune

pip install -r requirements.txt
```

### Full Training

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
accelerate launch --config_file "configs/fsdp_config.yaml" --main_process_port 7309 sft/finetune.py \
--model_name_or_path "models/DeepSeek-Coder/6.7b" \
--data_path "data/train/Mixed-Python-92k.jsonl" \
--max_train_samples 1024 \
--pad_mode "pack" \
--max_length 4096 \
--output_dir "weights/full" \
--overwrite_output_dir True \
--gradient_checkpointing True \
--num_train_epochs 3 \
--per_device_train_batch_size 16 \
--gradient_accumulation_steps 4 \
--learning_rate 5e-5 \
--lr_scheduler_type "cosine" \
--warmup_steps 100 \
--save_strategy "no" \
--logging_strategy "epoch" \
--bf16 True \
--tf32 True > logs/full.log 2>&1
```

### LoRA Training

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
accelerate launch --config_file "configs/fsdp_config_peft.yaml" --main_process_port 7309 sft/finetune.py \
--model_name_or_path "models/DeepSeek-Coder/6.7b" \
--use_peft_lora True \
--data_path "data/train/Mixed-Python-92k.jsonl" \
--max_train_samples 1024 \
--pad_mode "pack" \
--max_length 4096 \
--output_dir "weights/lora" \
--overwrite_output_dir True \
--gradient_checkpointing True \
--num_train_epochs 3 \
--per_device_train_batch_size 16 \
--gradient_accumulation_steps 4 \
--learning_rate 5e-5 \
--lr_scheduler_type "cosine" \
--warmup_steps 100 \
--save_strategy "no" \
--logging_strategy "epoch" \
--bf16 True \
--tf32 True > logs/lora.log 2>&1
```

### QLoRA Training

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
accelerate launch --config_file "configs/fsdp_config_qlora.yaml" --main_process_port 7309 sft/finetune.py \
--model_name_or_path "models/DeepSeek-Coder/6.7b" \
--use_peft_lora True \
--use_4bit_quantization True \
--bnb_4bit_quant_storage_dtype "bfloat16" \
--data_path "data/train/Mixed-Python-92k.jsonl" \
--max_train_samples 1024 \
--pad_mode "pack" \
--max_length 4096 \
--output_dir "weights/qlora" \
--overwrite_output_dir True \
--gradient_checkpointing True \
--num_train_epochs 3 \
--per_device_train_batch_size 16 \
--gradient_accumulation_steps 4 \
--learning_rate 5e-5 \
--lr_scheduler_type "cosine" \
--warmup_steps 100 \
--save_strategy "no" \
--logging_strategy "epoch" \
--bf16 True \
--tf32 True > logs/qlora.log 2>&1
```

## Reference

- [FSDP from Transformers Docs](https://huggingface.co/docs/transformers/fsdp)
- [Use PEFT and FSDP](https://huggingface.co/docs/peft/v0.15.0/en/accelerate/fsdp)
- [FSDP-QLoRA](https://huggingface.co/docs/bitsandbytes/v0.45.4/en/fsdp_qlora)
- [accelerate-issue2761](https://github.com/huggingface/accelerate/issues/2761#issuecomment-2142461407)
- [trl-issue1723](https://github.com/huggingface/trl/issues/1723#issuecomment-2269305410)
