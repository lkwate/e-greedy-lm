import pytorch_lightning as pl
from transformers import (
    EncoderDecoderModel,
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
)
import click
from loguru import logger
import torch
from .language_modelling import RLLMLightningModule
from .dataset import *
from .utils import action_table_from_file
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import os


os.environ["TOKENIZERS_PARALLELISM"] = "true"
DATASET_DIC = {
    "squad": SQuADLightningDataModule,
    "multi_news": MultiNewsLightningDataModule,
}


@click.command()
@click.argument("model_name", type=str)
@click.argument("action_table_file", type=click.Path(exists=True))
@click.argument("dataset_name", type=str)
@click.argument("log_dir", type=click.Path())
@click.option("--batch_size", type=int, default=32)
@click.option("--num_workers", type=int, default=4)
@click.option("--max_length", type=int, default=512)
@click.option("--learning_rate", type=float, default=1e-5)
@click.option("--k", type=int, default=10)
@click.option("--epsilon", type=float, default=0.2)
@click.option("--beta", type=float, default=0.06)
@click.option("--variance_type", type=str, default="local")
@click.option("--lr_factor", type=float, default=0.1)
@click.option("--lr_patience", type=int, default=4)
@click.option("--early_stopping_patience", type=int, default=5)
@click.option("--optimizer_name", type=str, default="Adam")
@click.option("--max_epochs", type=int, default=10)
@click.option("--val_check_interval", type=float, default=0.25)
@click.option("--accumulate_grad_batches", type=int, default=1)
@click.option("--save_top_k", type=int, default=5)
@click.option("--strategy", type=str, default="ddp", help="ddp, ddp_spawn ...")
@click.option("--random_seed", type=int, default=2021)
def main(
    model_name: str,
    action_table_file: str,
    dataset_name: str,
    log_dir: str,
    batch_size: int,
    num_workers: int,
    max_length: int,
    learning_rate: float,
    k: int,
    epsilon: float,
    beta: float,
    variance_type: str,
    lr_factor: float,
    lr_patience: int,
    early_stopping_patience: int,
    optimizer_name: str,
    max_epochs: int,
    val_check_interval: float,
    accumulate_grad_batches: int,
    save_top_k: int,
    strategy: str,
    random_seed: int,
):
    pl.seed_everything(random_seed, workers=True)

    logger.info("Actions table creation...")
    action_table = action_table_from_file(action_table_file, k)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if dataset_name not in DATASET_DIC:
        logger.error(f"Dataset {dataset_name} not available")

    logger.info(f"{dataset_name} lightning data module creation...")
    pl_data_module = DATASET_DIC[dataset_name](
        tokenizer, batch_size, num_workers, max_length
    )

    logger.info("Sequence-2-Sequence model building...")
    encoder = AutoModel.from_pretrained(model_name)
    decoder = AutoModelForCausalLM.from_pretrained(model_name, is_decoder=True)
    model = EncoderDecoderModel(encoder=encoder, decoder=decoder)

    pl_model = RLLMLightningModule(
        model,
        action_table,
        tokenizer,
        learning_rate,
        k,
        epsilon,
        beta,
        variance_type,
        lr_factor,
        lr_patience,
        optimizer_name,
    )

    trainer_config = {
        "max_epochs": max_epochs,
        "default_root_dir": log_dir,
        "val_check_interval": val_check_interval,
        "accumulate_grad_batches": accumulate_grad_batches,
        "strategy": strategy,
    }
    if torch.cuda.is_available():
        trainer_config["gpus"] = -1

    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=early_stopping_patience, verbose=False, strict=True
    )
    model_checkpoint_callback = ModelCheckpoint(
        dirpath=log_dir,
        filename="{epoch}-{val_loss:.4f}",
        monitor="val_loss",
        save_top_k=save_top_k,
    )

    trainer_config["callbacks"] = [early_stopping_callback, model_checkpoint_callback]
    pl_trainer = pl.Trainer(**trainer_config)

    logger.info("Training starts...")
    pl_trainer.fit(pl_model, datamodule=pl_data_module)
    logger.info("Training completed.")

    logger.info("Testing starts....")
    pl_trainer.test(pl_model, datamodule=pl_data_module)


if __name__ == "__main__":
    main()
