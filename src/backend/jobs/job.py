from rq import get_current_job, Queue, Connection
import redis
import time

def create_job(contentImage, styleImage):
    job = get_current_job()
    job.meta['image'] = contentImage
    job.save_meta()

    time.sleep(2)
    return True

def get_job(job_id):
    # TODO: Move redisURL to a config file
    redisURL = 'redis://redis:6379/0'
    with Connection(redis.from_url(redisURL)):
        q = Queue()
        job = q.fetch_job(job_id)
    return job