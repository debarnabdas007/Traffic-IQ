# Traffic-IQ: Edge AI Vehicle Analytics

## Overview

Traffic-IQ is a computationally efficient, real-time edge AI vehicle analytics platform. Instead of relying on heavy, resource-intensive Deep Learning models (like YOLO), this project systematically breaks down the problem using Classical Computer Vision for isolation and lightweight Machine Learning for classification. It is deployed as a responsive microservice using FastAPI, making it highly suitable for resource-constrained edge devices.

* * *

## Architecture Breakdown

### 1\. The "Vision" Side (OpenCV)

OpenCV handles the heavy lifting in the initial part of the pipeline, manipulating and analyzing video data before it reaches the Machine Learning layer. This prepares the scene so the classifier only has to evaluate the most important elements.

-   **Background Subtraction (MOG2):** Isolates moving vehicles by tracking the static background (roads) across multiple frames and subtracting it. This leaves behind bright, shifting "blobs" representing motion. Parameters like `history` are tuned to handle moving shadows or dynamic lighting.
    
-   **Morphological Operations:** Fixes "fragmentation" caused by dark windows or colors blending with the road.
    
    -   **Dilation:** Expands the boundaries of the white blobs, connecting detached pieces of a single vehicle.
        
    -   **Closing:** Fills in the holes within those shapes, ensuring the system sees one solid vehicle instead of detached parts.
        
-   **Bounding Boxes:** Draws a definitive box around each solid blob, extracting the physical dimensions needed for classification (x/y position, width, and height).
    

### 2\. The "Brains" Side (Machine Learning)

Once OpenCV isolates the moving objects, it passes their physical measurements to a Scikit-Learn Machine Learning pipeline.

-   **Feature Extraction:** The model relies on two primary physical features: **Area** (bounding box size) and **Aspect Ratio** (width divided by height).
    
-   **Feature Scaling:** A `StandardScaler` normalizes these features. Since Area operates on a vastly larger scale than Aspect Ratio, scaling ensures the sheer size of a vehicle doesn't overwhelm the subtle shape differences during calculation.
    
-   **K-Nearest Neighbors (KNN) Classification:** When a new vehicle is detected, the KNN algorithm compares its scaled features against labeled training data to predict whether the shape belongs to a car, truck, or bike.
    
-   **Heuristic Guards:** A hardcoded safety net prevents impossible physical predictions. For example, if a bounding box exceeds a certain area threshold, the system will automatically override the KNN prediction and classify it as a truck, preventing glitches or distance-based misclassifications.
    

### 3\. The "Engine" Side (FastAPI Backend)

The project is built as a fully deployable microservice, architected to handle continuous video streams without freezing the server.

-   **Generator Pattern:** Pauses the processing loop using Python's `yield` statement to send the current frame (image buffer) to the browser, then instantly resumes for the next frame.
    
-   **Motion JPEG (MJPEG) Streaming:** Streams a continuous sequence of JPEG images over the web using the `multipart/x-mixed-replace` media type. The browser replaces the old image so quickly it creates a smooth video feed.
    
-   **Decoupled API Endpoints:** \* `/video_feed`: Constantly streams the heavy video data.
    
    -   `/stats`: A lightweight, independent endpoint allowing the frontend to fetch current vehicle counts and JSON data without interfering with the video stream.
        

* * *

## Why Classical CV instead of Deep Learning (YOLO)?

Deep Learning provides fantastic accuracy but demands significant processing power, typically requiring dedicated GPUs for real-time inference. If deployed on edge hardware—like a Raspberry Pi or an old server rack by a highway—a massive neural network would drop to just a few frames per second.

By leveraging **Classical Computer Vision techniques (MOG2)** and **lightweight Machine Learning (KNN)**, Traffic-IQ is computationally inexpensive. It trades a small degree of semantic understanding for a massive gain in speed and efficiency, allowing it to run smoothly and reliably on standard, low-cost hardware.