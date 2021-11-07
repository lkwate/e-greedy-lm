from transformers import AutoTokenizer, AutoModel
import click
from utils import create_action


@click.command()
@click.argument("model_name", type=str)
@click.argument("output_file", type=click.Path())
@click.option("--k", type=int, default=10)
@click.option("--factor", type=int, default=10)
@click.option("--name", type=str, default="name")
def main(
    model_name: str, output_file: str, k: int = 10, factor: int = 10, name: str = "name"
):
    embedding = AutoModel.from_pretrained(model_name).embeddings.word_embeddings.weight
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    create_action(
        embedding, tokenizer, output_file, k=k, factor=factor, name=(name == "name")
    )


if __name__ == "__main__":
    main()
