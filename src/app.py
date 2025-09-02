from __future__ import annotations
import typer
from .flows.new_project_flow import run_new_project
from .flows.iterate_flow import run_iterate


app = typer.Typer(help="CrewAI Python Dev Starter CLI")


@app.command("new")
def new_project(
    prompt: str = typer.Option(
        ..., "--prompt", help="Short natural-language product prompt"
    ),
    out: str = typer.Option(..., "--out", help="Output directory for the new project"),
):
    """Greenfield: generate a brand new project from a short prompt."""
    run_new_project(prompt, out)


@app.command("iterate")
def iterate(
    prompt: str = typer.Option(
        ..., "--prompt", help="Change request for existing repo"
    ),
    repo: str = typer.Option(..., "--repo", help="Path to existing repository"),
):
    """Iterate on an existing project: targeted changes with tests and knowledge updates."""
    run_iterate(prompt, repo)


@app.command("fmt")
def fmt(repo: str = typer.Option(..., "--repo", help="Path to repo")):
    import subprocess
    import shlex

    subprocess.run(f"black {shlex.quote(repo)}", shell=True, check=False)


@app.command("lint")
def lint(repo: str = typer.Option(..., "--repo", help="Path to repo")):
    import subprocess
    import shlex

    subprocess.run(f"ruff {shlex.quote(repo)}", shell=True, check=False)


@app.command("test")
def test(repo: str = typer.Option(..., "--repo", help="Path to repo")):
    import subprocess

    subprocess.run("pytest -q", shell=True, cwd=repo, check=False)


if __name__ == "__main__":
    app()
