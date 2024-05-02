import tensorflow as tf


# tf.keras.metrics.F1Score does not support binary classification, so custom implementation was used
@tf.keras.saving.register_keras_serializable(package="lib", name="F1Score")
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        precision_result = self.precision.result()
        recall_result = self.recall.result()
        return 2 * ((precision_result * recall_result) / (
                precision_result + recall_result))

    def reset_state(self):
        self.precision.reset_states()
        self.recall.reset_states()
