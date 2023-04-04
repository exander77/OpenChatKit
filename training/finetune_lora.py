import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

DIR = os.path.dirname(os.path.abspath(__file__))

save_dir = f"${DIR}/../pretrained/Pythia-6.9B-deduped/EleutherAI_pythia-6.9b-deduped/"

model_name = 'EleutherAI/pythia-6.9b-deduped'

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map='auto',
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# for param in model.parameters():
#     param.requires_grad = False  # freeze the model - train adapters later
#     if param.ndim == 1:
#         # cast the small parameters (e.g. layernorm) to fp32 for stability
#         param.data = param.data.to(torch.float32)
#
# model.gradient_checkpointing_enable()  # reduce number of stored activations
# model.enable_input_require_grads()
#
#
# class CastOutputToFloat(nn.Sequential):
#     def forward(self, x): return super().forward(x).to(torch.float32)
#
#
# model.lm_head = CastOutputToFloat(model.lm_head)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)


import transformers
from datasets import load_dataset

data = load_dataset("/content/OpenChatKit/data/OIG/files")
data = data.map(lambda samples: tokenizer(samples['text']), batched=True)

trainer = transformers.Trainer(
    model=model,
    train_dataset=data['train'],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        max_steps=200,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir='outputs'
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

model.save_pretrained(save_directory=save_dir)

# Inference

import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

peft_model_id = model_name
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_8bit=True,
                                             device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)

batch = tokenizer("<human>: What is a computer?\n<bot>: ", return_tensors='pt')

with torch.cuda.amp.autocast():
    output_tokens = model.generate(**batch, max_new_tokens=256)

print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))
