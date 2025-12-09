
import typer
from typer.testing import CliRunner

app = typer.Typer()

@app.command()
def hello(name: str):
    print(f"Hello {name}")

runner = CliRunner()

def test_hello():
    print("Invoking with ['hello', 'World']")
    result = runner.invoke(app, ["hello", "World"])
    print(f"Exit code: {result.exit_code}")
    print(f"Output: {result.output}")

    print("\nInvoking with ['World']")
    result = runner.invoke(app, ["World"])
    print(f"Exit code: {result.exit_code}")
    print(f"Output: {result.output}")

if __name__ == "__main__":
    test_hello()
