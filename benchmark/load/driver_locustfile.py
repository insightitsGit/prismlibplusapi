"""
Locust load file for the PrismDriver two-node benchmark.

Set DRIVER_ENDPOINT env var to "/driver/baseline" or "/driver/query"
to control which path is under test.
"""

import os
import random
from locust import HttpUser, task, between

ENDPOINT = os.getenv("DRIVER_ENDPOINT", "/driver/baseline")

QUERIES = [
    "premium electronics item",
    "compact kitchen appliance",
    "deluxe sports equipment",
    "smart home device",
    "eco garden tool",
    "ultra lite clothing",
    "advanced automotive part",
    "classic book collection",
    "pro health supplement",
    "toy set for children",
    "budget furniture item",
    "wireless audio device",
    "outdoor camping gear",
    "fitness tracker watch",
    "ergonomic office chair",
    "portable solar charger",
    "organic skincare product",
    "gaming peripheral device",
    "kitchen storage solution",
    "children educational toy",
]


class DriverUser(HttpUser):
    wait_time = between(0.05, 0.2)

    @task
    def query(self):
        text = random.choice(QUERIES)
        self.client.post(ENDPOINT, params={"text": text}, name=ENDPOINT)
