from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import io, base64
import google.generativeai as genai
import os

app = Flask(__name__)
df = pd.read_csv("cleaned_ssi_data.csv")  # Default data

@app.route('/')
def homepage():
    return render_template("index.html")

@app.route('/dashboard')
def dashboard():
    summary_stats = df.describe().to_html(classes="table table-striped")
    correlation = df.corr(numeric_only=True)
    return render_template("dashboard.html", summary_stats=summary_stats, correlation=correlation.to_html())

@app.route('/hypothesis-testing')
def hypothesis_testing():
    small = df[df['Hospital_Category_RiskAdjustment'] == 'Smaller hospitals (<250 beds)']['SIR'].dropna()
    large = df[df['Hospital_Category_RiskAdjustment'] == 'Larger hospitals (>=250 beds)']['SIR'].dropna()
    t_stat, p_val = stats.ttest_ind(large, small, equal_var=False)
    conclusion = "significant" if p_val < 0.05 else "not significant"
    return jsonify({
        "t_stat": round(t_stat, 4),
        "p_val": round(p_val, 4),
        "conclusion": conclusion
    })

@app.route('/recommendations', methods=["POST"])
def generate_recommendations():
    context = request.json.get("user_context", "")
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return jsonify({"error": "GEMINI_API_KEY not found in environment variables."}), 400

    genai.configure(api_key=api_key)
    high_sir = df.groupby("Operative_Procedure")["SIR"].mean().sort_values(ascending=False).head(3)
    top_procedures = ", ".join(high_sir.index.tolist())

    prompt = (
        "You are a seasoned public health policy analyst. Based on the data insights below, "
        "In 300 words, generate 5 clear, simple and practical recommendations to reduce the Standardized Surgical Infection Ratio (SIR) "
        "across California hospitals:\n\n"
        f"Highest SIRs observed in procedures: {top_procedures}.\n\n"
        f"User Context: {context if context else 'No additional context provided.'}\n\n"
        "Keep recommendations simple, realistic, relevant, evidence-informed and avoid ambiguous words."
    )

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return jsonify({"recommendations": response.text})

if __name__ == "__main__":
    app.run(debug=True)
