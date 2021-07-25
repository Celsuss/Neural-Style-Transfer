from flask import Flask, request, make_response, jsonify
from rq import Queue, Connection
import redis
import os

from main import create_task

supported_types = ['jpg', 'png'] 
app = Flask(__name__) 

from flask_cors import CORS

cors = CORS(app)

app.config['CORS_HEADERS'] = 'Content-Type'
app.config['SECRET_KEY'] = os.urandom(24)

@app.route('/')
def test():
    return "Neural Style Transfer v1"

@app.route('/test', methods=['GET'])
def test_get():
    return 'Just a test!'

def run_task(contentFile, styleFile):
    # Tell RQ what Redis connection to use
    redis_conn = redis.Redis()
    q = Queue(connection=redis_conn)  # no args implies the default queue
    job = q.enqueue(create_task, contentFile, styleFile)
    return job.get_id()

@app.route('/post_images', methods=['POST'])
def post_images():
    # POST
    print('Post images')
    
    files = request.files.to_dict()

    if 'content_file' not in files or 'style_file' not in files:
        return make_response(jsonify({'status': 'fail', "msg": "Content or style file missing"}), 400)

    contentFile = files['content_file']
    styleFile = request.files['style_file']

    # jobId = run_task(contentFile, styleFile)

    res = make_response(jsonify({"status": "SUCCESS", "data": {"job_id": jobId}, "msg": "Files uploaded"}), 202)
    return res


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

