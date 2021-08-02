from flask import Flask, request, make_response, jsonify
from rq import Queue, Connection
from flask_cors import CORS
from PIL import Image
import redis
import io
import os

from jobs.job import create_job, test_print

supported_types = ['jpg', 'png'] 
app = Flask(__name__) 

cors = CORS(app)

app.config['CORS_HEADERS'] = 'Content-Type'
app.config['SECRET_KEY'] = os.urandom(24)

@app.route('/')
def test():
    return "Neural Style Transfer v1"

"""Add job to queue"""
def run_job(contentFile, styleFile):
    # TODO: Move redisURL to a config file
    redisURL = 'redis://redis:6379/0'
    with Connection(redis.from_url(redisURL)):
        q = Queue()
        job = q.enqueue(test_print)
        # job = q.enqueue(create_job, contentFile, styleFile)
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
    styleFile = files['style_file']

    if contentFile.filename.split('.')[-1].lower() not in supported_types or styleFile.filename.split('.')[-1].lower() not in supported_types:
        return make_response(jsonify({'error': 'fail', "msg": "Unsuported file type"}), 400)

    contentFile = contentFile.read()
    styleFile = styleFile.read()
    contentImage = Image.open(io.BytesIO(contentFile))
    styleImage = Image.open(io.BytesIO(styleFile))

    job = run_job(contentImage, styleImage)

    response_object = {
        "status": "success",
        "msg": "Images uploaded",
        "data": {
            "job_id": job.get_id()
        },
    }
    res = make_response(jsonify(response_object), 202)
    return res

"""Get Status from job"""
@app.route('/jobs/get_job_status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    # GET
    print('Get job status')
    test_print()

    return make_response(jsonify({'status': 'success', 'msg': 'Job status retrieved: {}'.format(job_id)}), 202)

    # TODO: Move redisURL to a config file
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

