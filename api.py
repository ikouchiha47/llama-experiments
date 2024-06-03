#!/usr/bin/env python

from flask import Flask, Response, request, abort
from flask_cors import CORS
import logging

from deps import Deps, ProblemData

app = Flask(__name__)
CORS(app, origins=["*"], expose_headers=["Content-Type"])

deps = Deps()

logging.basicConfig(level=logging.DEBUG)


@app.route("/")
def index():
    return "Hello, World!"


@app.route("/api/litkode/ingest/<problem_tag>", methods=["post"])
def ingest(problem_tag):
    data = request.get_json()
    obj = ProblemData(problem_tag, data["title"], data["problems"])
    if deps.add_problem_set(obj):
        return "problem set added"

    return "failed to add problem", 400


@app.route("/api/litkode/infer/<problem_tag>", methods=["get"])
def infer(problem_tag):
    query = request.args.get("query")

    if not query:
        abort(422)

    try:
        ps = deps.get_problem_set(problem_tag)
        logging.info(f"query {query}")

        result = ps.infer(query)
        logging.debug("Result", result)
        return result
    except Exception:
        logging.exception("failed to infer shit")
        return "failed to infer shit", 500


@app.route("/api/litkode/stream/<problem_tag>", methods=["get"])
def stream(problem_tag):
    query = request.args.get("query")

    if not query:
        abort(422)

    try:
        ps = deps.get_problem_set(problem_tag)
        # logging.info(f"query {query}")

        def generate():
            for chunk in ps.stream(query):
                yield chunk

        # resp = Response(generate(), content_type="text/plain")
        # resp.headers['X-Accel-Buffering'] = 'no'
        # return resp
        return Response(generate(), content_type="text/plain")

    except Exception:
        logging.exception("failed to infer shit")
        return "failed to infer shit", 500


if __name__ == "__main__":
    app.run(debug=True)
