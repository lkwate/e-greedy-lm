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
from .utils import action_table_from_file, build_model
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import os
import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "true"
DATASET_DIC = {
    "squad": SQuADLightningDataModule,
    "multi_news": MultiNewsLightningDataModule,
}


@click.command()
@click.argument("model_name", type=str)
@click.argument("action_table_file", type=click.Path(exists=True))
@click.argument("dataset_name", type=str)
@click.option("--batch_size", type=int, default=32)
@click.option("--num_workers", type=int, default=4)
@click.option("--max_length", type=int, default=512)
@click.option("--learning_rate", type=float, default=1e-5)
@click.option("--k", type=int, default=10)
@click.option("--epsilon", type=float, default=0.2)
@click.option("--beta", type=float, default=0.06)
@click.option("--add_variance", is_flag=True)
@click.option("--variance_type", type=str, default="local")
@click.option("--lr_factor", type=float, default=0.1)
@click.option("--lr_patience", type=int, default=4)
@click.option("--optimizer_name", type=str, default="Adam")
@click.option("--checkpoint_path", type=click.Path(exists=True))
@click.option("--output_file", type=click.Path(exists=False))
@click.option("--split", type=click.Choice(["train", "eval", "test"]), default="eval")
@click.option("--limit_batches", type=int, default=-1)
@click.option("--full_model", is_flag=True)
def main(
    model_name: str,
    action_table_file: str,
    dataset_name: str,
    batch_size: int,
    num_workers: int,
    max_length: int,
    learning_rate: float,
    k: int,
    epsilon: float,
    beta: float,
    add_variance: bool,
    variance_type: str,
    lr_factor: float,
    lr_patience: int,
    optimizer_name: str,
    checkpoint_path: str,
    output_file: str,
    split: str,
    limit_batches: int,
    full_model: bool,
):

    logger.info("Actions table creation...")
    action_table = action_table_from_file(action_table_file, k)

    if dataset_name not in DATASET_DIC:
        logger.error(f"Dataset {dataset_name} not available")

    logger.info("Sequence-2-Sequence model building...")
    model, tokenizer = build_model(model_name, full_model)

    logger.info(f"{dataset_name} lightning data module creation...")
    pl_data_module = DATASET_DIC[dataset_name](
        tokenizer, batch_size, num_workers, max_length
    )
    pl_data_module.prepare_data()
    if split == "train":
        data_module = pl_data_module.train_dataloader()
    elif split == "eval":
        data_module = pl_data_module.val_dataloader()
    elif split == "test":
        data_module = pl_data_module.test_dataloader()

    pl_model = RLLMLightningModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        action_table=action_table,
        tokenizer=tokenizer,
        learning_rate=learning_rate,
        k=k,
        epsilon=epsilon,
        beta=beta,
        variance_type=variance_type,
        lr_factor=lr_factor,
        lr_patience=lr_patience,
        optimizer_name=optimizer_name,
        add_variance=True,
    )
    pl_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pl_model = pl_model.to(device)

    logger.info("Start the evaluation...")
    limit_batches = float("inf") if limit_batches == -1 else limit_batches
    with torch.no_grad():
        with open(output_file, "w") as of:
            n_batches = 0
            for batch in tqdm.tqdm(data_module, desc="Save in %s ..." % output_file):
                # batch["input_ids"], batch["attention_mask"], batch["labels"], batch["decoder_attention_mask"]
                x = batch["encoder_input_ids"].to(device)
                y = pl_model.generate(x)
                for x_i, y_i in zip(x, y):
                    of.writelines(
                        [
                            "input : %s\n"
                            % tokenizer.decode(x_i, skip_special_tokens=True),
                            "output : %s\n\n"
                            % tokenizer.decode(y_i, skip_special_tokens=True),
                        ]
                    )
                n_batches += 1
                if n_batches > limit_batches:
                    break


if __name__ == "__main__":
    main()
