import streamlit as st
import requests
import base64
import io
from PIL import Image, ImageDraw, ImageFont
import glob
from base64 import decodebytes
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt

import asyncio
import logging
import logging.handlers
import queue
import threading
import urllib.request
from pathlib import Path
from typing import List, NamedTuple
import time

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

import av
import cv2
from aiortc.contrib.media import MediaPlayer

from streamlit_webrtc import (
    AudioProcessorBase,
    ClientSettings,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

### setting up the logger to log messages created by the script to help debug in main()
logger = logging.getLogger(__name__)

#### For the webrtc plug in, this is setting the default values for the widget,
#### incluing removing the audio
WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
)

def main():
    """
    Main part of the script that defines the main method and calls the logo detection.
    Also sets up a logger for debugging purposes.
    """
    st.header("Real-time Streamlit Logo Detection with Roboflow")

    image_dashboard()

    logger.debug("=== Alive threads ===")
    for thread in threading.enumerate():
        if thread.is_alive():
            logger.debug(f" {thread.name} ({thread.ident})")

    ##########
    ##### Set up sidebar.
    ##########

    # Add in location to select image.

def image_dashboard():
    st.sidebar.write('#### Select an image to upload.')
    uploaded_file = st.sidebar.file_uploader('',
                                            type=['png', 'jpg', 'jpeg'],
                                            accept_multiple_files=False)

    st.sidebar.write('[Find this dataset and more on Roboflow Universe.](https://universe.roboflow.com/)')

    ## Add in sliders.
    confidence_threshold = st.sidebar.slider('Confidence threshold: What is the minimum acceptable confidence level for displaying a bounding box?', 0.0, 1.0, 0.5, 0.01)
    overlap_threshold = st.sidebar.slider('Overlap threshold: What is the maximum amount of overlap permitted between visible bounding boxes?', 0.0, 1.0, 0.5, 0.01)


    image = Image.open('./images/roboflow_full_logo_color.png')
    st.sidebar.image(image,
                    use_column_width=True)

    image = Image.open('./images/streamlit_logo.png')
    st.sidebar.image(image,
                    use_column_width=True)

    ##########
    ##### Set up main app.
    ##########

    ## Title.
    st.write('# Face Detection Model built with Roboflow')

    ## Pull in default image or user-selected image.
    if uploaded_file is None:
        # Default image.
        default_img = './images/demo-faces.jpg'
        image = Image.open(default_img)

    else:
        # User-selected image.
        image = Image.open(uploaded_file)

    ## Subtitle.
    st.write('### Inferenced Image')

    # Convert to JPEG Buffer.
    buffered = io.BytesIO()
    image.save(buffered, quality=90, format='JPEG')

    # Base 64 encode.
    img_str = base64.b64encode(buffered.getvalue())
    img_str = img_str.decode('ascii')

    ### Setting up the url and query parameters for roboflow endpoint
    # the overlap and confidence query parameters are set within the
    ## Construct the URL to retrieve image.
    upload_url = ''.join([
        'https://detect.roboflow.com/face-detection-mik1i/5',
        f'?api_key={st.secrets["api_key"]}',
        '&format=image',
        f'&overlap={overlap_threshold * 100}',
        f'&confidence={confidence_threshold * 100}',
        '&stroke=2',
        '&labels=True'
    ])

    ## POST to the API.
    r = requests.post(upload_url,
                    data=img_str,
                    headers={
        'Content-Type': 'application/x-www-form-urlencoded'
    })

    image = Image.open(BytesIO(r.content))

    # Convert to JPEG Buffer.
    buffered = io.BytesIO()
    image.save(buffered, quality=90, format='JPEG')

    # Display image.
    st.image(image,
            use_column_width=True)

    ## Construct the URL to retrieve JSON.
    upload_url = ''.join([
        'https://detect.roboflow.com/face-detection-mik1i/5',
        f'?api_key={st.secrets["api_key"]}'
    ])

    ## POST to the API.
    r = requests.post(upload_url,
                    data=img_str,
                    headers={
        'Content-Type': 'application/x-www-form-urlencoded'
    })

    ## Save the JSON.
    output_dict = r.json()

    # Map detected classes to uniquely colored bounding boxes
    color_map = {
        "face": "#D41159",
    }

    ## Generate list of confidences.
    confidences = [box['confidence'] for box in output_dict['predictions']]

    ## Summary statistics section in main app.
    st.write('### Summary Statistics')
    st.write(f'Number of Bounding Boxes (ignoring overlap thresholds): {len(confidences)}')
    st.write(f'Average Confidence Level of Bounding Boxes: {(np.round(np.mean(confidences),4))}')

    ## Histogram in main app.
    st.write('### Histogram of Confidence Levels')
    fig, ax = plt.subplots()
    ax.hist(confidences, bins=10, range=(0.0,1.0))
    st.pyplot(fig)

    ## Display the JSON in main app.
    st.write('### JSON Output')
    st.write(r.json())


### the main program where we call the main method
if __name__ == "__main__":
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    main()
