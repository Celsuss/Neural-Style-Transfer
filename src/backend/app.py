from flask import Flask, request, make_response, jsonify
from rq import Queue, Connection
from flask_cors import CORS
from PIL import Image
import redis
import io
import os

from jobs.job import create_job, get_job

supported_types = ['jpg', 'png'] 
app = Flask(__name__) 

cors = CORS(app)

app.config['CORS_HEADERS'] = 'Content-Type'
app.config['SECRET_KEY'] = os.urandom(24)

@app.route('/')
def test():
    return 'Neural Style Transfer v1'

"""Add job to queue"""
def run_job(contentFile, styleFile):
    # TODO: Move redisURL to a config file
    redisURL = 'redis://redis:6379/0'
    with Connection(redis.from_url(redisURL)):
        q = Queue()
        job = q.enqueue(create_job, contentFile, styleFile)
        return job

"""Upload images and start a job to create a new image"""
@app.route('/post_images', methods=['POST'])
def post_images():
    # POST
    print('Post images')
    
    files = request.files.to_dict()

    if 'content_file' not in files or 'style_file' not in files:
        return make_response(jsonify({'error': 'fail', 'msg': 'Content or style file missing'}), 400)

    contentFile = files['content_file']
    styleFile = files['style_file']

    if contentFile.filename.split('.')[-1].lower() not in supported_types or styleFile.filename.split('.')[-1].lower() not in supported_types:
        return make_response(jsonify({'error': 'fail', 'msg': 'Unsuported file type'}), 400)

    contentFile = contentFile.read()
    styleFile = styleFile.read()
    contentImage = Image.open(io.BytesIO(contentFile))
    styleImage = Image.open(io.BytesIO(styleFile))

    job = run_job(contentImage, styleImage)

    response_object = {
        'status': 'success',
        'msg': 'Images uploaded',
        'data': {
            'job_id': job.get_id()
        },
    }
    res = make_response(jsonify(response_object), 202)
    return res

from base64 import encodebytes

"""Get Status from job"""
@app.route('/jobs/get_job_status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    # GET
    print('Get job status for job: {}'.format(job_id))

    job = get_job(job_id)
    if job:
        response_code = 200
        response_object = {
            'status': 'success',
            'data': {
                'job_id': job.get_id(),
                'job_status': job.get_status(),
                'job_result': job.result,
            },
        }
    else:
        response_code = 400
        response_object = {'status': 'error', 'msg': 'Job not found'}

    return make_response(jsonify(response_object), response_code)

"""Get image from job"""
@app.route('/jobs/get_job_image_url/<job_id>', methods=['GET'])
def get_job_image_url(job_id):
    # GET
    print('Get job image url for job: {}'.format(job_id))

    job = get_job(job_id)
    if job:
        job_image = job.meta['image']
        
        path = 'static/{}.png'.format(job_id)
        job_image.save(path)

        response_code = 200
        response_object = {
            'status': 'success',
            'data': {
                'image_url': path
            },
        }

    else:
        response_code = 400
        response_object = {'status': 'error', 'msg': 'Job not found'}

    return make_response(jsonify(response_object), response_code)  

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

