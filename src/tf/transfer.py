# https://www.youtube.com/watch?v=4oNdaQk0Qv4
import tensorflow as tf
import tensorflow.feature_column as fc
import tensorflow_hub as hub
import numpy as np

def transfer_learn():
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
    review = hub.text_embedding_column(
        "review", module_url)

    features = {
        "review": np.array(["a masterpiece", "ineditable shoe leather"])
    }

    labels = np.array([[1], [0]])

    input_fn = tf.estimator.inputs.numpy_input_fn(features, labels, shuffle=True )

    hidden_units = [1]
    estimator = tf.estimator.DNNClassifier(
        hidden_units,
        feature_columns=[review])

    estimator.train(input_fn, max_steps=10)
    accuracy_score = estimator.evaluate(input_fn=input_fn)["accuracy"]
    print("\nTest Accuracy: {0:f}%\n".format(accuracy_score*100))
    # estimator.predict()



transfer_learn()
