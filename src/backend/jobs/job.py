from rq import get_current_job
import time

def create_job(contentImage, styleImage):
    job = get_current_job()
    job.meta['image'] = contentImage
    job.save_meta()

    time.sleep(5)
    return True

def test_print():
    print('test print')
    return "work done"