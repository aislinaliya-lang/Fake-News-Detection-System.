from flask import Flask, render_template, request
from model import predict_news

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    text = ""

    if request.method == "POST":

        text = request.form.get("news_text", "").strip()

        if text:
            prediction, confidence = predict_news(text)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        text=text
    )


if __name__ == "__main__":
    app.run(debug=True)
