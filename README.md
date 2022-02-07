# AI Support for building detection
This repository contains code for creating and using a U-Net model for the task of detecting buildings from orthophotos.

## Environment requirements
The code should run on python 3.9.9 and a separate environment for this application is recommended.
After installing the environment and cloning the repository, install the packages from requirements.txt. If using pip, the following line will then install the necessary libraries in the right versions:
```
pip install -r requirements.txt
```

It is recommended to use a GPU when performing inference on new orhthophotos and training new models. If a GPU is installed, but not correctly, the application might exit abruptly. In this case, and if the user wants to use CPU instead of GPU, the following line can be added to the beginning of inference.py and training.py:
```
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

## Inference on new images
* constants.py contains settings and folders that the application will use
* make sure there are some image(s) in the "ORTHOPHOTOS" folder given in constants.py
* the "MODEL_NAME" variable in constants.py holds the name of the model that will be used during inference
* run "python inference.py" from the command line
  - first the orthophoto(s) are sliced into 512x512 squares, and preprocessing is performed on the slices
  - secondly, this performs inference on the sliced orthophoto(s), and connects the slices back to the original size
* the produced binary image file(s) that are the layer of predicted buildings will be created in "INFERENCE_OUTPUT" folder given in constants.py

## Training a new model
It is possible to train a new model based on labeled data. The model will train on the input orthophoto(s) and corresponding ground truth binary image(s), and generate a model that can be used for inference on new orthophotos. The structure of the model will be the same for all the models, but giving the model more data to train on, could improve the accuracy of the model.

* constants.py contains settings and folders that the application will use
* make sure there are some orthophoto(s) in the "ORTHOPHOTOS" and "ORTHOPHOTOS_MASKS" folders. The name of the orthophoto and orthophoto masks shall be the same to make the application match the orthophoto(s) with the corresponding ground truth (masks)
* the "MODEL_NAME" variable in constants.py holds the name of the model that will be created during training
* run "python training.py" from the command line
  - this trains a new model with the given data and will store the model in the "MODEL_SAVE_FOLDER" folder in constants.py
 
