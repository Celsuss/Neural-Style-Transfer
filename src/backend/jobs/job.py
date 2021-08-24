from rq import get_current_job, Queue, Connection
import redis
import time

from jobs.styleTransfer import styleTransfer

"""Create a Neural Style Transfer Job"""
def create_job(contentImage, styleImage):
    result_image = styleTransfer(contentImage, styleImage)
    job = get_current_job()
    job.meta['image'] = result_image
    job.save_meta()
    return True

"""Returns the job with the provided ID"""
def get_job(job_id):
    # TODO: Move redisURL to a config file
    redisURL = 'redis://redis:6379/0'
    with Connection(redis.from_url(redisURL)):
        q = Queue()
        job = q.fetch_job(job_id)
    return job

def test_job(contentImage, styleImage):
    job = get_current_job()
    job.meta['image'] = contentImage
    job.save_meta()

    time.sleep(2)
    return True