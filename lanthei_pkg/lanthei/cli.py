import os
import sys
from pathlib import Path
from openai import OpenAI

CONFIG_PATH = Path.home() / ".lanthei_key.txt"

def get_api_key():
    if CONFIG_PATH.exists():
        return CONFIG_PATH.read_text().strip()
    key = input("Enter your Gemini API key (get one free at aistudio.google.com): ").strip()
    CONFIG_PATH.write_text(key)
    print(f"Saved. (stored locally at {CONFIG_PATH})")
    return key

def main():
    api_key = get_api_key()
    client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Ask me anything: ")

    try:
        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": question}],
        )
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()