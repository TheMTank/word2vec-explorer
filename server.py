#!/usr/bin/env python

import sys
import os
import json
import decimal

from flask import Flask, request, render_template, send_from_directory, jsonify
#from flask_restful import Resource, Api, reqparse

from explorer import Model

STATIC_DIR = os.path.dirname(os.path.realpath(__file__)) + '/public'
CACHE = {}

class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, decimal.Decimal):
            return float(o)
        return super(DecimalEncoder, self).default(o)

app = Flask(__name__)

@app.route("/")
def main():
    return render_template('index.html')

@app.route('/js/<path:path>')
def send_js(path):
    return send_from_directory('public/js', path)

@app.route('/styles/<path:path>')
def send_styles(path):
    return send_from_directory('public/styles', path)

@app.route("/api/explore")
def explore():
    # query='cat'
    query = request.args.get('query', '')
    limit = request.args.get('limit', '1000')
    enable_clustering = 'True'
    num_clusters = request.args.get('num_clusters', '30')

    cache_key = '-'.join([query, limit, enable_clustering, num_clusters])
    result = CACHE.get(cache_key, {})
    if len(result) > 0:
        return jsonify({'result': CACHE[cache_key], 'cached': True})
    try:
        exploration = model.explore(query, limit=int(limit))
        exploration.reduce()
        if len(enable_clustering):
            if (len(num_clusters)):
                num_clusters = int(num_clusters)
            exploration.cluster(num_clusters=num_clusters)
        result = exploration.serialize()
        CACHE[cache_key] = result
        return jsonify({'result': result,
                        'cached': False})
    except KeyError:
        return jsonify({'error': {'message': 'No vector found for ' + query}})


@app.route("/api/compare")
def compare():
    limit = request.args.get('limit', 100)
    queries = request.args.getlist('queries[]')
    print(limit)
    print(queries)
    try:
        result = model.compare(queries, limit=int(limit))
        return jsonify({'result': result})
    except KeyError:
        return jsonify({'error':
                            {'message': 'No vector found for {}'.format(queries)}})


if __name__ == '__main__':
    # python explore.py embeddings_object
    if len(sys.argv) < 2:
        sys.stderr.write('Usage: {} <Model file>\n'.format(sys.argv[0]))
        sys.exit()

    model = Model(sys.argv[1])

    app.run(host='0.0.0.0', port=8080)
