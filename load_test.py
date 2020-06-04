import ray

ray.init()


@ray.remote
def predict(bytesio, endpoint):
    import requests
    resp = requests.post(f"http://localhost:8000/{endpoint}", data=bytesio)
    print(resp.text)
    return resp.json()


cat = open("cat.png", "rb").read()
place = open("plate-b58bps.jpg", "rb").read()

while True:
    ray.get([predict.remote(cat, "pipeline") for _ in range(8)] +
            [predict.remote(place, "pipeline") for _ in range(6)])
