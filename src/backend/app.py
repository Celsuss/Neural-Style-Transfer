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

"""Add task to queue"""
def run_task(contentFile, styleFile):
    # Tell RQ what Redis connection to use
    redis_conn = redis.Redis()
    q = Queue(connection=redis_conn)  # no args implies the default queue
    job = q.enqueue(create_task, contentFile, styleFile)
    return job

"""Upload images and start a task to create a new image"""
@app.route('/post_images', methods=['POST'])
def post_images():
    # POST
    print('Post images')
    
    files = request.files.to_dict()

    if 'content_file' not in files or 'style_file' not in files:
        return make_response(jsonify({'error': 'fail', "msg": "Content or style file missing"}), 400)

    contentFile = files['content_file']
    styleFile = request.files['style_file']

    # task = run_task(contentFile, styleFile)

    response_object = {
        "status": "success",
        "msg": "Images uploaded",
        "data": {
            "task_id": "0" # task.get_id()
        },
    }
    res = make_response(jsonify(response_object), 202)
    return res

"""Get Status from job"""
@app.route('/jobs/get_job_status', methods=['GET'])
def get_job_status(task_id):
    # GET
    print('Get task status')

    redisURL = 'redis://redis:6379/0'
    with Connection(redis.from_url(redisURL)):
        q = Queue()
        task = q.fetch_job(task_id)
    if task:
        response_code = 200
        response_object = {
            "status": "success",
            "data": {
                "task_id": task.get_id(),
                "task_status": task.get_status(),
                "task_result": task.result,
            },
        }
    else:
        response_code = 400
        response_object = {"status": "error"}

    return make_response(jsonify(response_object), response_code)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

