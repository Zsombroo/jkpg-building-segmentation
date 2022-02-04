''' AI Support for building detection
    Copyright (C) 2022  Zsombor Toth

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
'''


import tensorflow as tf

from image_slicing_helper import extract_file_names_from_input_data
from ortho_slicing import slice_images
from tf_ds_gen import load_training_data
from u_net_model import create_model

from constants import ENSEMBLE_SIZE
from constants import EPOCHS
from constants import IMAGE_SHAPE
from constants import MODEL_NAME
from constants import MODEL_SAVE_FOLDER
from constants import X_PATH


slice_images()

extract_file_names_from_input_data(X_PATH)

performances = []
for i in range(ENSEMBLE_SIZE):
    print(f'{MODEL_NAME}_{i} / {ENSEMBLE_SIZE}')
    train, val, test = load_training_data()
    model = create_model(f'{MODEL_NAME}_{i}')
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            mode='min',
            verbose=1,
            restore_best_weights=True,
            patience=20,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            f'{MODEL_SAVE_FOLDER}/{MODEL_NAME}_{i}.h5',
            monitor='val_loss',
            mode='min',
            verbose=1,
            save_best_only=True,
        )
    ]
    history = model.fit(
        train,
        epochs=EPOCHS,
        validation_data=val,
        callbacks=callbacks
    )
    performance = model.evaluate(test)
    performances.append(performance)

models = [tf.keras.models.load_model(f'{MODEL_SAVE_FOLDER}/{MODEL_NAME}_{i}.h5') for i in range(ENSEMBLE_SIZE)]
model_input = tf.keras.Input(shape=IMAGE_SHAPE)
model_outputs = [model(model_input) for model in models]
ensemble_output = tf.keras.layers.Average()(model_outputs)
ensemble_model = tf.keras.Model(inputs=model_input, outputs=ensemble_output, name=MODEL_NAME)
ensemble_model.save(f'{MODEL_SAVE_FOLDER}/{MODEL_NAME}.h5')
