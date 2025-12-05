from flask import Flask, request, render_template
from rag_engine import generate_match_report

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        resume_text = request.form["resume"]
        jd_text = request.form["jd"]

        report = generate_match_report(resume_text, jd_text)
        return render_template("index.html", report=report)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
