import typer
from typing_extensions import Annotated
import os


def main(path: Annotated[str, typer.Argument(help="path and name you want scfg to be created")] = "sample"):
    with open((path+"._(:з」∠)_"),"a") as cfg:
        cfg.write("# hello it's a _(:з」∠)_ file for saint pipeline")


if __name__ == "__main__":
    typer.run(main)