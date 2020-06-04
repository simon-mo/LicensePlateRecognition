import ray
from ray import serve


class ComposedModel:
    def __init__(self):
        self.detector_handle = serve.get_handle("object")
        self.alpr_handle = serve.get_handle("alpr")

    async def __call__(self, flask_request):
        image_data = flask_request.data

        object_found = await self.detector_handle.remote(data=image_data)

        if object_found["label"] != "car":
            return {"contains_car": False}
        if object_found["score"] < 0.4:
            return {"contain_car": False}

        license_plate = await self.alpr_handle.remote(data=image_data)
        return {"contains_car": True, "license_plate": license_plate}


ray.init(address="auto")
serve.init()

serve.create_endpoint("composed", "/composed", methods=["POST"])
serve.create_backend("composed:v1", ComposedModel, config={
    "num_replicas": 2
})
serve.set_traffic("composed", {"composed:v1": 1})




