import typer

from train import train
from predict import app as predict_app

app = typer.Typer()
app.command(name="train")(train)
app.add_typer(predict_app, name="predict")

if __name__ == "__main__":
    app()
