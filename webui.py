import streamlit as st
import requests
from pprint import pformat


def predict(bytesio, endpoint):
    resp = requests.post(
        f"http://localhost:8000/{endpoint}", data=bytesio.read())
    print(resp.text)
    return resp.json()


st.markdown("# Model Serving with Ray Serve")

image = st.file_uploader("Upload Image")

if image:
    st.image(image, width=600)

    options = ("Object Detector", "License Plate Recognizer", "Pipeline 🎉")
    endpoints = ("object", "alpr", "composed")

    st.info("Choose your models:")
    chosen = st.selectbox("Choose your models:", options)

    for opt, ep in zip(options, endpoints):
        if chosen == opt:
            st.markdown(f"### {opt}")
            image.seek(0)

            if opt.startswith("Pipeline"):
                st.graphviz_chart("""
digraph G {

  image -> object_detector;
  object_detector -> license_recognizer [label="contains car"];
  object_detector -> output [label="no car "];
  license_recognizer -> output;

}
                """)

            if st.button("Predict!", key=opt + "-button"):
                with st.spinner("Pinging Ray Serve HTTP Endpoint..."):
                    result = predict(image, ep)
                st.markdown("```\n" + pformat(result) + "\n```\n")
