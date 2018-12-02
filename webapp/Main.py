

from flask import Flask, render_template
app = Flask('meme-1')
@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__": 
    app.run("0.0.0.0")