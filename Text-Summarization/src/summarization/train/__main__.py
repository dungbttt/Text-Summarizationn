from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch
import wandb
import pytorch_lightning as pl
from pathlib import Path
from rouge import Rouge
import datasets
from transformers import AdamW, T5ForConditionalGeneration, T5TokenizerFast as T5Tokenizer

from ..dataset.datamodule import *
from ..model.model import *
import argparse

# PARSERS
parser = argparse.ArgumentParser()
parser.add_argument(
    '--max_epochs', '-me', type=int, default=2, help='max epoch')
parser.add_argument(
    '--batch_size', '-bs', type=int, default=8, help='batch size')
parser.add_argument(
    '--num_workers', '-nw', type=int, default=4, help='number of workers')
parser.add_argument(
    '--wandb', '-w', default=False, action='store_true', help='use wandb or not')
parser.add_argument(
    '--wandb_key', '-wk', type=str, help='wandb API key')
args = parser.parse_args()

def train():
    pl.seed_everything(42)

    project_path = Path('../')
    data_path = project_path / 'data'
    data_path.mkdir(parents=True, exist_ok=True)

    #Train Data Path
    train_ds_path = data_path / 'train'
    train_ds_path.mkdir(parents=True, exist_ok=True)

    #Test Data Path
    test_ds_path = data_path / 'test'
    test_ds_path.mkdir(parents=True, exist_ok=True)

    #Val Data Path
    val_ds_path = data_path / 'val'
    val_ds_path.mkdir(parents=True, exist_ok=True)

    #Model Checkpoint
    chkpt_path = project_path / "checkpoints"
    chkpt_path.mkdir(parents=True, exist_ok=True)

    train_data = datasets.load_dataset("ccdv/cnn_dailymail", "3.0.0", split="train")   
    val_data = datasets.load_dataset("ccdv/cnn_dailymail", "3.0.0", split="validation")

    MODEL_NAME = 't5-base'
    N_EPOCHS = args.max_epochs
    BATCH_SIZE = args.batch_size

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, model_max_length=512)
    data_module = NewsSummaryDataModule(train_data, val_data, tokenizer, batch_size=BATCH_SIZE, num_workers=4)
    model = NewsSummaryModel()

    chkpt_path = "../checkpoints"
    checkpoint_callback = ModelCheckpoint(
        dirpath = str(chkpt_path),
        filename="Best-T5-{epoch}-{step}-{val_loss:.2f}",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
    )
    wandb.login(key=args.wandb_key)
    logger = WandbLogger(project="text-summarization", name="T5", log_model="all")

    lr_callback = LearningRateMonitor("step")

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, lr_callback],
        max_epochs=N_EPOCHS,
        enable_progress_bar=True,
        log_every_n_steps=500,
        val_check_interval=500
    )

    trainer.fit(model, data_module)
    
if __name__ == "__main__":
    train()