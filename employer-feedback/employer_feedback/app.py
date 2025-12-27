import os

from flask import (
    Flask,
    render_template,
    request,
    session,
    redirect,
    url_for,
)

from nlp_utils import analyze_feedback_list

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change_me")

USERS = {
    "admin": "password123",
}


def is_authenticated() -> bool:
    return session.get("user") is not None


@app.route("/login", methods=["GET", "POST"])
def login():
    if is_authenticated():
        return redirect(url_for("index"))

    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if USERS.get(username) == password:
            session["user"] = username
            return redirect(url_for("index"))

        error = "Invalid username or password."

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


@app.route("/", methods=["GET", "POST"])
def index():
    if not is_authenticated():
        return redirect(url_for("login"))

    analysis_result = None
    keywords = None
    error = None
    sentiment_stats = None

    if request.method == "POST":
        raw_text = request.form.get("feedback_text", "").strip()

        if not raw_text:
            error = "Please enter at least one feedback comment."
        else:
            feedback_list = raw_text.split("\n")
            df, top_keywords = analyze_feedback_list(feedback_list)

            if df.empty:
                error = "No valid feedback found."
            else:
                analysis_result = df.to_dict(orient="records")
                keywords = top_keywords

                total = len(df)
                pos = sum(df["sentiment"].str.lower() == "positive")
                neg = sum(df["sentiment"].str.lower() == "negative")
                neu = total - pos - neg

                sentiment_stats = {
                    "total": total,
                    "positive_count": pos,
                    "negative_count": neg,
                    "neutral_count": neu,
                    "positive_pct": round(pos * 100 / total, 2),
                    "negative_pct": round(neg * 100 / total, 2),
                    "neutral_pct": round(neu * 100 / total, 2),
                }

    return render_template(
        "index.html",
        analysis_result=analysis_result,
        keywords=keywords,
        error=error,
        sentiment_stats=sentiment_stats,
    )


if __name__ == "__main__":
    app.run(debug=True)
