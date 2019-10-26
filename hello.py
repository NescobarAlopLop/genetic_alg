from flask import Flask, flash, redirect, render_template, request, session, abort
from bokeh.embed import server_document


app = Flask(__name__)


@app.route("/")
def hello():
    script = server_document("http://127.0.0.1:5006/bokeh-sliders")
    return render_template('hello.html', bokS=script)

@app.route("/test")
def test():
    script = server_document("http://0.0.0.0:5006/bokeh-sliders")
    return render_template('hello.html', bokS=script)


if __name__ == "__main__":
    # app.run(host='0.0.0.0')
    app.run(host='localhost')
