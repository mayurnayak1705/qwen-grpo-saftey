from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model
from reward_function import reward_safety

import wandb


# --------------------------------
# Dataset
# --------------------------------

raw_dataset = load_dataset(
    "PKU-Alignment/PKU-SafeRLHF-prompt"
)

small_dataset = (
    raw_dataset["train"]
    .shuffle(seed=42)
    .select(range(1500))
)

split_dataset = small_dataset.train_test_split(
    test_size=0.1,
    seed=42
)

train_data = split_dataset["train"]
test_data = split_dataset["test"]

# print(train_data)
# print(test_data)
# print(train_data.column_names)
# print(train_data[0])


# --------------------------------
# Model
# --------------------------------

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    torch_dtype="auto",
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-0.5B"
)


# --------------------------------
# LoRA
# --------------------------------

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
)

model = get_peft_model(
    model,
    lora_config
)

model.print_trainable_parameters()


# --------------------------------
# W&B
# --------------------------------

wandb.login()

wandb.init(
    project="GRPO-qwen-saftey"
)


# --------------------------------
# GRPO config
# --------------------------------

training_args = GRPOConfig(
    output_dir="GRPO",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    max_prompt_length=512,
    max_completion_length=96,
    num_generations=4,
    optim="adamw_torch",
    num_train_epochs=1,
    bf16=True,
    report_to=["wandb"],
    remove_unused_columns=False,
    logging_steps=1,
    push_to_hub=True,
    hub_model_id="mayurnayak1705/qwen-0.5b-grpo-safety",
)


# --------------------------------
# Trainer
# --------------------------------

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_safety],
    args=training_args,
    train_dataset=train_data,
)


# --------------------------------
# Train
# --------------------------------

trainer.train()
trainer.push_to_hub()