# This script provides a command-line interface (CLI) to interact with the Google Gemini AI
# model through the OpenAI API facade. It allows users to:
# - Chat with Gemini (default command or `chat`)
# - Explain the contents of a specified code file (`explain`)
# - Fix or modify a specified code file based on an instruction (`fix`), with diff review and Git commit integration.
# It stores the Gemini API key locally for convenience.

import os
import sys
import argparse
import difflib
from pathlib import Path
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()
CONFIG_PATH = Path.home() / ".lanthei_key.txt"


def get_api_key():
    if CONFIG_PATH.exists():
        return CONFIG_PATH.read_text().strip()
    key = input("Enter your Gemini API key (get one free at aistudio.google.com): ").strip()
    CONFIG_PATH.write_text(key)
    console.print(f"[green]Saved.[/green] (stored locally at {CONFIG_PATH})")
    return key


def get_client():
    api_key = get_api_key()
    return OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )


def ask_gemini(client, prompt, model="gemini-2.5-flash"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def cmd_chat(args):
    client = get_client()
    question = " ".join(args.question) if args.question else input("Ask me anything: ")
    try:
        console.print(ask_gemini(client, question))
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


def cmd_explain(args):
    filepath = Path(args.file)
    if not filepath.exists():
        console.print(f"[red]File not found: {filepath}[/red]")
        return

    code = filepath.read_text()
    client = get_client()
    prompt = f"Explain what this code does, in plain language:\n\n{code}"
    try:
        console.print(ask_gemini(client, prompt))
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


def cmd_fix(args):
    filepath = Path(args.file)
    if not filepath.exists():
        console.print(f"[red]File not found: {filepath}[/red]")
        return

    original_code = filepath.read_text()
    instruction = " ".join(args.instruction)
    client = get_client()

    prompt = (
        f"Here is a Python file:\n\n{original_code}\n\n"
        f"Task: {instruction}\n\n"
        "Return ONLY the complete corrected file content, with no explanation, "
        "no markdown code fences, just the raw file contents."
    )

    console.print("[yellow]Asking Gemini for changes...[/yellow]")
    try:
        new_code = ask_gemini(client, prompt)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return

    new_code = new_code.strip()
    if new_code.startswith("```"):
        lines = new_code.split("\n")
        new_code = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

    diff = list(difflib.unified_diff(
        original_code.splitlines(keepends=True),
        new_code.splitlines(keepends=True),
        fromfile="before",
        tofile="after",
    ))

    if not diff:
        console.print("[yellow]No changes suggested.[/yellow]")
        return

    diff_text = "".join(diff)
    console.print(Panel(Syntax(diff_text, "diff", theme="ansi_dark"), title="Proposed Changes"))

    console.print("\n[bold green][A][/bold green] Apply & Commit   [bold red][R][/bold red] Reject")
    choice = input("Choose: ").strip().lower()

    if choice == "a":
        filepath.write_text(new_code)
        os.system(f'git add "{filepath}"')
        os.system(f'git commit -m "AI edit: {instruction}"')
        console.print(f"[green]Applied and committed changes to {filepath}[/green]")
    else:
        console.print("[yellow]Rejected. No changes made.[/yellow]")


def main():
    known_commands = {"explain", "fix", "chat"}

    if len(sys.argv) > 1 and sys.argv[1] in known_commands:
        parser = argparse.ArgumentParser(prog="lanthei")
        subparsers = parser.add_subparsers(dest="command")

        explain_parser = subparsers.add_parser("explain")
        explain_parser.add_argument("file")
        explain_parser.set_defaults(func=cmd_explain)

        fix_parser = subparsers.add_parser("fix")
        fix_parser.add_argument("file")
        fix_parser.add_argument("instruction", nargs="+")
        fix_parser.set_defaults(func=cmd_fix)

        chat_parser = subparsers.add_parser("chat")
        chat_parser.add_argument("question", nargs="*")
        chat_parser.set_defaults(func=cmd_chat)

        args = parser.parse_args()
        args.func(args)
    else:
        chat_args = argparse.Namespace(question=sys.argv[1:])
        cmd_chat(chat_args)


if __name__ == "__main__":
    main()