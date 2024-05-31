#!/usr/bin/env python

from flask import Flask, request

app = Flask(__name__)


@app.route("/")
def index():
    return "Hello, World!"


@app.route("/v1/litkode/ingest/<problem_tag>", methods=["post"])
def ingest(problem_tag):
    data = request.get_json()

    return "hello, world!"


@app.route("/v1/litkode/inference/<problem_tag>", methods=["get"])
def infer(problem_tag):
    return "hello, world!"


if __name__ == "__main__":
    app.run(debug=True)
