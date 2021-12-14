import typer

from predict import app as predict_app
from train import train

app = typer.Typer()
app.command(name="train")(train)
app.add_typer(predict_app, name="predict")

if __name__ == "__main__":
    app()
