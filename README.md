# AI Support for building detection
This repository contains code for creating and using a U-Net model for the task of detecting buildings from orthophotos.

## Environment requirements
The code should run on python 3.9.9 and a separate environment for this application is recommended.
After installing the environment and cloning the repository, install the packages from requirements.txt. If using pip, the following line will then install the necessary libraries in the right versions:
```
pip install -r requirements.txt
```

## Inference on new images
* constants.py contains settings and folders that the application will use
* make sure there are some image(s) in the "ORTHOPHOTOS" folder given in constants.py
* run "python ortho_slicing.py" from the command line
  - this slices the image(s) into 512x512 squares, and performs preprocessing on the slices
* run "python inference.py" from the command line
  - this performs inference on the sliced image(s), and connects the slices back to the original size
* the produced binary image file(s) that are the layer of predicted buildings will be created in "INFERENCE_OUTPUT" folder given in constants.py

## Training a new model
It is possible to train a new model based on labeled data. The model will train on the corresponding input orthophoto(s) and ground truth binary image(s), and generate a model that can be used for inference on new orthophotos. The structure of the model will be the same for all the models, but giving the model more data to train on, could improve the accuracy of the model.

* constants.py contains settings and folders that the application will use
* make sure there are some image(s) in the "ORTHOPHOTOS" and "ORTHOPHOTOS_MASKS" folders. The name of the orthophoto and orthophoto masks shall be the same to make the application match the orthophoto(s) with the corresponding ground truth (masks)
* run "python training.py" from the command line
  - this trains a new model with the given data and will store the model in the "MODEL_SAVE_FOLDER" folder in constants.py
 
