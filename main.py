import typer

from train import train

app = typer.Typer()
app.command(name="train")(train)

if __name__ == "__main__":
    app()
