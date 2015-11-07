import datetime
import time


class GoogleMapsClient(object):
    """3rd party maps client; we CANT EDIT THIS."""

    def __init__(self):
        self.requests_made = 0

    def make_request(self):
        self.requests_made += 1
        now = datetime.datetime.now().time()
        return "%d - %s - San Francisco" % (self.requests_made, now)


class GeneralRateLimitedClient(object):
    def __init__(self, rate, seconds=1):
        self.rate = rate  # messages
        self.per_seconds = seconds  # seconds
        self.allowance = rate
        self.last_check = time.time()

        self.third_party = GoogleMapsClient()

    def make_request(self):
        current = time.time()
        time_passed = current - self.last_check
        self.last_check = current
        self.allowance += time_passed * (self.rate / self.per_seconds)

        if self.allowance > self.rate:
            self.allowance = self.rate  # Up to the cap.

        if self.allowance < 1.0:
            # Discard the message
            time.sleep(1)
            return ''

        self.allowance -= 1.0
        return self.third_party.make_request()


class RateLimitedClient(object):
    def __init__(self, limit):
        self.requests_limit = limit  # requests limit per second.
        self.last_check = None
        self.requests_made = 0
        self.third_party = GoogleMapsClient()

    def make_request(self):
        now = time.time()  # In second.

        if not self.last_check or now - self.last_check > 1:
            self.last_check = now
            self.requests_made = 0

        self.requests_made += 1

        if self.requests_made > self.requests_limit:
            time.sleep(1)
            return ''  # Drop the request, instead of holding it.

        return self.third_party.make_request()

if __name__ == '__main__':
    client = RateLimitedClient(2)
    for _ in range(10):
        # 0, 1 sent
        # 2, drop and wait
        # 3, 4 sent
        # 5, drop and wait
        # 6, 7 sent
        # 8, drop and wait
        # 9 sent
        print client.make_request()
