
# SQL Finetuning

## Data Preparation

Run:

  

python scripts/prepare_sql.py

This will run the code that downloads the datasets and applies the tokenization. The default modeil is **checkpoints/meta-llama/Llama-2-7b-chat-hf/**. You can change the models by running:

  

python scripts/prepare_sql.py --checkpoint_dir <PATH-OF-YOUR-MODEL>

After running this script, you will fine two new files inside the folder **data/sql-create-context/**, called train.pt and test.pt.

## Model Finetuning

  

python finetune/lora.py --data_dir data/sql-create-context/ --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-hf/ --out_dir out/lora/sql_llama --precision bf16-mixed --quantize bnb.nf4

  

## Model Prompting

Prompting Llama2 is described [here](https://huggingface.co/blog/llama2#how-to-prompt-llama-2). The Base prompt for the base model is in the following form:
```
<s>[INST] <<SYS>>
{{ system_prompt }}
<</SYS>>

{{ user_message }} [/INST]
```
Multiple rounds of conversation are represented in the following way:
```
<s>[INST] <<SYS>>
{{ system_prompt }}
<</SYS>>

{{ user_msg_1 }} [/INST] {{ model_answer_1 }} </s><s>[INST] {{ user_msg_2 }} [/INST]
```
In the case of multiple rounds, it may be necessary to append the end tag manually, I didn't have the opportunity of checking.

The base system prompt is the following:

	You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. 
	Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
	Please ensure that your responses are socially unbiased and positive in nature.

	If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.
	If you don't know the answer to a question, please don't share false information.

### Example
```
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

There's a llama in my garden 😱 What should I do? [/INST]
```