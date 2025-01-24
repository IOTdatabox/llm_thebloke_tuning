import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import pandas as pd
import json
import logging
from datasets import Dataset
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_bitagent_data(row):
    """Convert BitAgent format to training format"""
    conversation = json.loads(row['conversation'])
    tools = json.loads(row['tools'])
    
    # Combine into a single prompt-response format
    prompt = f"<s>[INST]User Query: {conversation[0]['content']}\nAvailable Tools: {json.dumps(tools, indent=2)}[/INST]"
    response = f"[ASSISTANT]{conversation[-1]['content']}</s>"
    
    return {
        "text": prompt + response
    }

def prepare_glaive_data(row):
    """Convert Glaive format to training format"""
    system = row['system']
    chat = row['chat']
    
    prompt = f"<s>[INST]System: {system}\nChat: {chat}[/INST]"
    # Extract assistant response from chat
    response = "[ASSISTANT]" + chat.split("ASSISTANT:")[-1].strip() + "</s>"
    
    return {
        "text": prompt + response
    }

def prepare_bfcl_data(row):
    """Convert BFCL format to training format"""
    conversation = json.loads(row['conversation'])
    tools = json.loads(row['tools'])
    
    prompt = f"<s>[INST]Query: {conversation[0]['content']}\nAvailable Functions: {json.dumps(tools, indent=2)}[/INST]"
    response = f"[ASSISTANT]Function Call: {json.dumps(conversation[-1], indent=2)}</s>"
    
    return {
        "text": prompt + response
    }

def load_and_prepare_datasets(data_dir="bitagent.data/samples"):
    """Load and prepare all datasets"""
    datasets = []
    
    # Load BitAgent dataset
    try:
        bitagent_df = pd.read_csv(f"{data_dir}/bitagent_sample.csv")
        bitagent_data = [prepare_bitagent_data(row) for _, row in bitagent_df.iterrows()]
        datasets.extend(bitagent_data)
        logger.info(f"Loaded {len(bitagent_data)} BitAgent samples")
    except Exception as e:
        logger.error(f"Error loading BitAgent dataset: {str(e)}")

    # Load Glaive dataset
    try:
        glaive_df = pd.read_csv(f"{data_dir}/glaive_sample.csv")
        glaive_data = [prepare_glaive_data(row) for _, row in glaive_df.iterrows()]
        datasets.extend(glaive_data)
        logger.info(f"Loaded {len(glaive_data)} Glaive samples")
    except Exception as e:
        logger.error(f"Error loading Glaive dataset: {str(e)}")

    # Load BFCL dataset
    try:
        bfcl_df = pd.read_csv(f"{data_dir}/bfcl_sample.csv")
        bfcl_data = [prepare_bfcl_data(row) for _, row in bfcl_df.iterrows()]
        datasets.extend(bfcl_data)
        logger.info(f"Loaded {len(bfcl_data)} BFCL samples")
    except Exception as e:
        logger.error(f"Error loading BFCL dataset: {str(e)}")

    return Dataset.from_dict({"text": [d["text"] for d in datasets]})

def train():
    # Load model and tokenizer
    model_name = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=False
    )

    # Prepare dataset
    dataset = load_and_prepare_datasets()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=2e-5,
        fp16=True,
        gradient_checkpointing=True
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    # Train the model
    trainer.train()
    
    # Save the model
    output_dir = "./fine_tuned_model"
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")

if __name__ == "__main__":
    train()