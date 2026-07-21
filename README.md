# Lanthei CLI

A simple command-line tool that lets you chat with Google's Gemini AI directly from your terminal — using your own free API key.

## Install

pip install lanthei-cli

## Usage

Ask a question directly:

lanthei "explain what a python list comprehension is"

Or run it with no arguments for an interactive prompt:

lanthei

The first time you run it, you'll be asked to paste your Gemini API key (get one free at https://aistudio.google.com). It's saved locally at `~/.lanthei_key.txt` and only used on your machine — never sent anywhere except directly to Google's API.

## Requirements

- Python 3.9+
- A free Gemini API key from Google AI Studio

## License

MIT
