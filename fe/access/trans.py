import requests
from urllib.parse import urljoin

class Trans:
    def __init__(self, url_prefix):
        self.url_prefix = urljoin(url_prefix, "trans/")

    def trans_in(self, data: dict) -> int:
        json = data
        url = urljoin(self.url_prefix, "trans_in")
        r = requests.post(url, json=json)
        return r.status_code
    
    def get_result(self) -> (int, int):
        url = urljoin(self.url_prefix, "get_result")
        r = requests.post(url)
        response = r.json()
        return r.status_code, response.get("score")