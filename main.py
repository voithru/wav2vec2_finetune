import typer

from train import train
from predict import predict

app = typer.Typer()
app.command(name="train")(train)
app.command(name="predict")(predict)

if __name__ == "__main__":
    app()
