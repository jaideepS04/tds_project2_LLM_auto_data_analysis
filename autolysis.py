from fastapi import FastAPI
import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import openai
# Constants
OUTPUT_MARKDOWN = "README.md"
OUTPUT_IMAGES = ["chart1.png", "chart2.png", "chart3.png"]

# Set API token manually
openai.api_key = os.getenv("AIPROXY_TOKEN", "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjEwMDI0ODJAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.AoGx1fHFJVPj4Lu5O5uTM7gM_JT8m_9RbYSEjRdOhcI")

# Initialize the FastAPI application
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Autolysis API!"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "query": q}

# Load dataset
def load_dataset(filename):
    try:
        return pd.read_csv(filename, encoding="ISO-8859-1")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

# Analyze the dataset
def analyze_dataset(df):
    analysis = {
        "shape": df.shape,
        "columns": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "summary": df.describe(include='all').to_dict()
    }
    return analysis

# Generate visualizations
def generate_visualizations(df):
    charts = []

    if df.select_dtypes(include=['float64', 'int64']).shape[1] > 1:
        plt.figure(figsize=(8, 6))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.savefig(OUTPUT_IMAGES[0])
        charts.append(OUTPUT_IMAGES[0])
        plt.close()

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if not numeric_cols.empty:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[numeric_cols[0]], kde=True, bins=30)
        plt.title(f"Distribution of {numeric_cols[0]}")
        plt.savefig(OUTPUT_IMAGES[1])
        charts.append(OUTPUT_IMAGES[1])
        plt.close()

    categorical_cols = df.select_dtypes(include=['object']).columns
    if not categorical_cols.empty:
        plt.figure(figsize=(8, 6))
        sns.countplot(y=df[categorical_cols[0]])
        plt.title(f"Count of {categorical_cols[0]}")
        plt.savefig(OUTPUT_IMAGES[2])
        charts.append(OUTPUT_IMAGES[2])
        plt.close()

    return charts

# Summarize findings using LLM
def summarize_findings(df, analysis, charts):
    summary_prompt = (
        f"Dataset Analysis:\n"
        f"- Shape: {analysis['shape']}\n"
        f"- Columns: {list(analysis['columns'].keys())}\n"
        f"- Missing values: {analysis['missing_values']}\n"
        f"- Summary statistics: {analysis['summary']}\n"
        f"\nBased on this analysis, write a detailed story about the dataset, its patterns, and its implications. "
        f"Include references to visualizations: {', '.join(charts)}."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a data analysis assistant."},
                {"role": "user", "content": summary_prompt}
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error generating summary: {e}")
        return "Error generating summary."

# Write README file
def write_readme(summary, charts):
    with open(OUTPUT_MARKDOWN, "w") as f:
        f.write("# Automated Dataset Analysis\n\n")
        f.write(summary)
        f.write("\n\n## Visualizations\n")
        for chart in charts:
            f.write(f"![{chart}]({chart})\n")

# Main function
def main():
    if len(sys.argv) == 3 and sys.argv[1] == "run":
        dataset_file = sys.argv[2]

        df = load_dataset('load the path of file')

        analysis = analyze_dataset(df)
        charts = generate_visualizations(df)
        summary = summarize_findings(df, analysis, charts)

        write_readme(summary, charts)
        print("Analysis complete. Results saved in README.md and visualizations.")
    else:
        print("Usage: uv run autolysis.py dataset.csv")

if __name__ == "__main__":
    main()
