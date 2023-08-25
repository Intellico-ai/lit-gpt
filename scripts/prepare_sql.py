"""Implementation derived from https://github.com/tloen/alpaca-lora"""
import json
import sys
from pathlib import Path

import requests
import torch
from torch.utils.data import random_split
from tqdm import tqdm

from datasets import load_dataset

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.tokenizer import Tokenizer

DATA_FILE_URL = ""
DATA_FILE_NAME = "sql-create-context.json"
DESTINATION_PATH = Path("data/sql-create-context")
CHECKPOINT_DIR = Path("checkpoints/meta-llama/Llama-2-7b-chat-hf/")
TEST_SPLIT_FRACTION = 0.10
IGNORE_INDEX = -1
MASK_INPUTS = False  # as in alpaca-lora
HUGGING_FACE_DATASET_PATH = "b-mc2/sql-create-context"
SEED = 42


def prepare(
    destination_path: Path = DESTINATION_PATH,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    test_split_fraction: float = TEST_SPLIT_FRACTION,
    seed: int = SEED,
    mask_inputs: bool = MASK_INPUTS,
    data_file_name: str = DATA_FILE_NAME,
    data_file_url: str = DATA_FILE_URL,
    ignore_index: int = IGNORE_INDEX,
) -> None:
    """Prepare the SQL dataset for instruction tuning.

    The output is a training and test dataset saved as `train.pt` and `test.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """
    """Prepare the SQL dataset for instruction tuning.

    The output is a training and test dataset saved as `train.pt` and `test.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """
    with open(checkpoint_dir / "lit_config.json", "r") as file:
        config = json.load(file)
        max_seq_length = config["block_size"]
    print("Loading data file...")
    # this sql dataset has only a train set
    sql_dataset = load_dataset(HUGGING_FACE_DATASET_PATH)["train"]
    sql_dataset = sql_dataset.rename_column("question", "instruction")
    sql_dataset = sql_dataset.rename_column("context", "input")
    sql_dataset = sql_dataset.rename_column("answer", "output")

    destination_path.mkdir(parents=True, exist_ok=True)
    data_file_path = destination_path / data_file_name

    # download_if_missing(data_file_path, data_file_url)
    # with open(data_file_path, "r", encoding="utf-8") as file:
    #     data = json.load(file)

    print("Loading tokenizer...")
    tokenizer = Tokenizer(checkpoint_dir)

    # Partition the dataset into train and test
    train_set, test_set = random_split(
        sql_dataset,
        [1.0 - test_split_fraction, test_split_fraction],
        generator=torch.Generator().manual_seed(seed),
    )
    train_set, test_set = list(train_set), list(test_set)

    print(f"train has {len(train_set):,} samples")
    print(f"test has {len(test_set):,} samples")
    print("Processing train split ...")
    train_set = [
        prepare_sample(
            example=sample,
            tokenizer=tokenizer,
            max_length=max_seq_length,
            mask_inputs=mask_inputs,
            ignore_index=ignore_index,
        )
        for sample in tqdm(train_set)
    ]
    torch.save(train_set, destination_path / "train.pt")
    print("Processing test split ...")
    test_set = [
        prepare_sample(
            example=sample,
            tokenizer=tokenizer,
            max_length=max_seq_length,
            mask_inputs=mask_inputs,
            ignore_index=ignore_index,
        )
        for sample in tqdm(test_set)
    ]
    torch.save(test_set, destination_path / "test.pt")


def download_if_missing(file_path: Path, file_url: str):
    """Downloads the raw json data file and saves it in the given destination."""
    if file_path.exists():
        return
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(requests.get(file_url).text)


def prepare_sample(
    example: dict,
    tokenizer: Tokenizer,
    max_length: int,
    mask_inputs: bool = MASK_INPUTS,
    ignore_index: int = IGNORE_INDEX,
):
    """Processes a single sample.

    Each sample in the dataset consists of:
    - instruction: A string describing the task
    - input: A string holding a special input value for the instruction.
        This only applies to some samples, and in others this is empty.
    - output: The response string

    This function processes this data to produce a prompt text and a label for
    supervised training. The prompt text is formed as a single message including both
    the instruction and the input. The label/target is the same message but with the
    response attached.

    Finally, both the prompt and the label get tokenized. If desired, all tokens
    in the label that correspond to the original input prompt get masked out (default).
    """
    full_prompt = generate_prompt(example)
    full_prompt_and_response = full_prompt + example["output"]
    encoded_full_prompt = tokenizer.encode(full_prompt, max_length=max_length)
    encoded_full_prompt_and_response = tokenizer.encode(full_prompt_and_response, eos=True, max_length=max_length)

    # The labels are the full prompt with response, but with the prompt masked out
    labels = encoded_full_prompt_and_response.clone()
    if mask_inputs:
        labels[: len(encoded_full_prompt)] = ignore_index

    return {
        **example,
        "input_ids": encoded_full_prompt_and_response,
        "input_ids_no_response": encoded_full_prompt,
        "labels": labels,
    }


def generate_prompt(example):
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""

    system_prompt = (
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being"
        " safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or"
        " illegal content. Please ensure that your responses are socially unbiased and positive in nature. You are"
        " an expert SQL programmer, you can produce efficient and correct SQL code. The user will generally give"
        " you a context, which consist in table that compose the SQL database, and than a query. The SQL context"
        " can vary during the conversation, therefore the query that you will generate must consider all the"
        " changes in the database that you have been toldRespond truthfully and be concise. "
    )
    if example["input"]:
        template = (
            "[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\nThe SQL database is composed by the following sql tables:"
            " {sql_context}\n{user_request} [/INST]"
        )


        sql_create_context = example["input"]
        user_request= example["instruction"]
        formatted_string = template.format(system_prompt=system_prompt, sql_context=sql_create_context, user_request=user_request)
        return formatted_string
    else:
        template = "[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_request} [/INST]"
        user_request= example["instruction"]
        formatted_string = template.format(system_prompt=system_prompt, user_request=user_request)
        return formatted_string

if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
