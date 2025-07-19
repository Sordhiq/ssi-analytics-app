# ssi_dashboard_vercel.py

from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

html_form = """
    <h2>SSI Dashboard (Lightweight)</h2>
    <form method="post" enctype="multipart/form-data">
        <label>Upload CSV File:</label>
        <input name="file" type="file">
        <input type="submit">
    </form>
"""

@app.get("/", response_class=HTMLResponse)
def main():
    return html_form

@app.post("/", response_class=HTMLResponse)
async def analyze(file: UploadFile):
    df = pd.read_csv(file.file)
    response_html = "<h2>Summary Statistics</h2>"
    response_html += df.describe().to_html()

    # Correlation Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    img_html = f'<h2>Correlation Heatmap</h2><img src="data:image/png;base64,{encoded}"/>'
    response_html += img_html

    return HTMLResponse(content=response_html)
