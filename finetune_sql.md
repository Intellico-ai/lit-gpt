# SQL Finetuning
## Data Preparation
Run:

     python scripts/prepare_sql.py 
This will run the code that downloads the datasets and applies the tokenization. The default modeil is **checkpoints/meta-llama/Llama-2-7b-hf/**. You can change the models by running:

    python scripts/prepare_sql.py --checkpoint_dir <PATH-OF-YOUR-MODEL>
After running this script, you will fine two new files inside the folder **data/sql-create-context/**, called train.pt and test.pt.
## Model Finetuning

    python finetune/lora.py --data_dir data/sql-create-context/ --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-hf/ --out_dir out/lora/sql_llama --precision bf16-mixed --quantize bnb.nf4
