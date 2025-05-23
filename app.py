from flask import Flask, render_template, request, jsonify
import pandas as pd
from ner_model import extract_entities

app = Flask(__name__)

# Load news dataset
news_df = pd.read_csv("data/news.csv")

@app.route("/")
def index():
    return render_template("index.html", articles=news_df.to_dict(orient="records"))

@app.route("/article/<int:article_id>")
def article(article_id):
    article_data = news_df.iloc[article_id]
    return render_template(
        "article.html",
        title=article_data["Title"],
        content=article_data["Content"],
        article_id=article_id  # Pass article ID for AJAX request
    )

@app.route("/fetch_entities/<int:article_id>")
def fetch_entities(article_id):
    article_data = news_df.iloc[article_id]
    entities = extract_entities(article_data["Content"])
    
    return jsonify({
        "people": entities.get("People", []),
        "organizations": entities.get("Organizations", []),
        "locations": entities.get("Locations", [])
    })

# New route to render entity extraction page
@app.route("/extract", methods=["GET", "POST"])
def extract():
    entities = None
    if request.method == "POST":
        text = request.form["text"]
        if text.strip():
            entities = extract_entities(text)
    return render_template("extract.html", entities=entities)

if __name__ == "__main__":
    app.run(debug=True)
