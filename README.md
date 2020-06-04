## Automatic Vehicle + License Detector built on Ray Serve

Pre-req:

- Build and install darkflow from github https://github.com/thtrieu/darkflow
- Run `make download` to download the model weight

Running:

```
streamlit run webui.py # Display the interactive webui

# Start a ray cluster
ray up --head

# Deploy the models
python serve_basics.py
python serve_composed.py
```
