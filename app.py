import gradio as gr
import pandas as pd
from datasets import load_dataset
from openai import OpenAI
import spaces

print("Loading dataset...")
dataset = load_dataset("nyxtesla/lanthei")
splits = {name: dataset[name].to_pandas() for name in dataset.keys()}
print("Dataset loaded:", {name: len(df) for name, df in splits.items()})

def browse(split_name, search_text, num_rows):
    df = splits[split_name]
    if search_text.strip():
        df = df[df["text"].str.contains(search_text, case=False, na=False)]
    display_df = df.head(int(num_rows)).copy()
    display_df["label"] = display_df["label"].map({0: "negative", 1: "positive"})
    return display_df, f"{len(df)} matching rows found"

def label_counts(split_name):
    df = splits[split_name]
    counts = df["label"].value_counts().rename({0: "negative", 1: "positive"})
    return counts.to_frame(name="count")

@spaces.GPU
def analyze_review(review_text, user_api_key):
    if not user_api_key.strip():
        return "Please enter your own Gemini API key above first."
    if not review_text.strip():
        return "Paste or select a review's text above first."

    client = OpenAI(
        api_key=user_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    try:
        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[{
                "role": "user",
                "content": f"In 2-3 short sentences, explain why this movie review reads as positive or negative:\n\n{review_text}"
            }],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

with gr.Blocks(title="Lanthei Dataset Explorer") as demo:
    gr.Markdown("# Lanthei Dataset Explorer\nBrowse the dataset, then optionally analyze a review with your own Gemini key.")

    with gr.Row():
        split_dropdown = gr.Dropdown(choices=list(splits.keys()), value="train", label="Split")
        search_box = gr.Textbox(label="Search review text (optional)", placeholder="e.g. amazing")
        rows_slider = gr.Slider(minimum=5, maximum=100, value=20, step=5, label="Rows to show")

    search_button = gr.Button("Search")
    status_text = gr.Markdown()
    results_table = gr.Dataframe(headers=["text", "label"], wrap=True)

    gr.Markdown("### Label distribution for selected split")
    counts_table = gr.Dataframe()

    gr.Markdown("---\n### Analyze a review with AI (your key is used only for this request, never stored)")
    api_key_box = gr.Textbox(label="Your Gemini API key", type="password", placeholder="AIza...")
    review_input = gr.Textbox(label="Paste a review's text here", lines=4)
    analyze_button = gr.Button("Analyze with AI")
    analysis_output = gr.Markdown()

    search_button.click(fn=browse, inputs=[split_dropdown, search_box, rows_slider], outputs=[results_table, status_text])
    split_dropdown.change(fn=label_counts, inputs=[split_dropdown], outputs=[counts_table])
    analyze_button.click(fn=analyze_review, inputs=[review_input, api_key_box], outputs=[analysis_output])

if __name__ == "__main__":
    demo.launch()