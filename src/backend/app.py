from flask import Flask, request, make_response, jsonify
from rq import Queue, Connection
import redis
import os

from main import create_job

supported_types = ['jpg', 'png'] 
app = Flask(__name__) 

from flask_cors import CORS

cors = CORS(app)

app.config['CORS_HEADERS'] = 'Content-Type'
app.config['SECRET_KEY'] = os.urandom(24)

@app.route('/')
def test():
    return "Neural Style Transfer v1"

"""Add job to queue"""
def run_job(contentFile, styleFile):
    # Tell RQ what Redis connection to use
    redis_conn = redis.Redis()
    q = Queue(connection=redis_conn)
    job = q.enqueue(create_job, contentFile, styleFile)
    return job

"""Upload images and start a job to create a new image"""
@app.route('/post_images', methods=['POST'])
def post_images():
    # POST
    print('Post images')
    
    files = request.files.to_dict()

    if 'content_file' not in files or 'style_file' not in files:
        return make_response(jsonify({'error': 'fail', "msg": "Content or style file missing"}), 400)

    contentFile = files['content_file']
    styleFile = request.files['style_file']

    # job = run_job(contentFile, styleFile)

    response_object = {
        "status": "success",
        "msg": "Images uploaded",
        "data": {
            "job_id": "0" # job.get_id()
        },
    }
    res = make_response(jsonify(response_object), 202)
    return res

"""Get Status from job"""
@app.route('/jobs/get_job_status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    # GET
    print('Get job status')

    return make_response(jsonify({'status': 'success', 'msg': 'Job status retrieved: {}'.format(job_id)}), 202)

    redisURL = 'redis://redis:6379/0'
    with Connection(redis.from_url(redisURL)):
        q = Queue()
        job = q.fetch_job(job_id)
    if job:
        response_code = 200
        response_object = {
            "status": "success",
            "data": {
                "job_id": job.get_id(),
                "job_status": job.get_status(),
                "job_result": job.result,
            },
        }
    else:
        response_code = 400
        response_object = {"status": "error"}

    return make_response(jsonify(response_object), response_code)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

