import sys
import os
from datetime import datetime
import tensorflow as tf
from tensorflow import keras

sys.path.append(os.path.join(os.path.dirname(__file__), "compression-master", "models"))

import bls2017

# Define the Keras TensorBoard callback.
logdir="logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

model = bls2017.BLS2017Model(0.01 ,128)
model.compile()

print(model.layers)

# tf.keras.utils.plot_model(model.analysis_transform, to_file="model_bls2017_analysis.png", show_shapes=True, expand_nested=True, show_layer_names=True, show_layer_activations=True)
# tf.keras.utils.plot_model(model.synthesis_transform, to_file="model_bls2017_synthesis.png", show_shapes=True, expand_nested=True, show_layer_names=True, show_layer_activations=True)

# tf.keras.utils.plot_model(model.prior, to_file="model_bls2017_prior.png", show_shapes=True, expand_nested=True, show_layer_names=True, show_layer_activations=True)
# # Save the model summary as a string
# model_summary = []
# model.summary(print_fn=lambda x: model_summary.append(x))
# model_summary = "\n".join(model_summary)

# # Write the model summary to TensorBoard
# with tf.summary.create_file_writer(logdir).as_default():
#     tf.summary.graph(model_summary)

# Write the model graph to TensorBoard
# with tf.summary.create_file_writer(logdir).as_default():
#     tf.summary.trace_on(graph=True, profiler=True)
#     # Run a forward pass to generate the graph
#     dummy_input = tf.ones((1, 128, 128, 3))  # Replace input_size with the actual input size
#     model(dummy_input)
#     # Save the graph to TensorBoard
#     tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=logdir)