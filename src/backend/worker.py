
from rq import Connection, Worker
import redis

redisURL = 'redis://redis:6379/0'

# Provide queue names to listen to as arguments to this script,
# similar to rq worker
def start_worker():
    redisConnection = redis.from_url(redisURL)

    with Connection(redisConnection):
        queues = ['default']
        worker = Worker(queues)
        worker.work()

    return 0

if __name__ == '__main__':
    start_worker()