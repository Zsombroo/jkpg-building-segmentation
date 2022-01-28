# jkpg-building-segmentation / AI Support for building detection

## Environment requirements
The code should run on python 3.9.9 and a separate environment for this application is recommended.
If using pip, the following line will then install the necessary libraries in the right versions:
```
pip install -r requirements.txt
```

## Inference on new images
* constants.py contains settings and folders that the application will use
* make sure there are some image(s) in the "ORTHOPHOTOS" folder given in constants.py
* run "python ortho_slicing.py" from the command line
* run "python inference.py" from the command line
* the produced binary image file(s) that are the layer of predicted buildings will be created in "INFERENCE_OUTPUT" folder given in constants.py

