from keras import backend as K
from keras import metrics
import tensorflow as tf
from keras.utils.np_utils import to_categorical


class Adversarial(object):

    def __init__(self):
        self.sess = K.get_session()
        K.set_session(self.sess)

    def model_loss(self, y, model, mean=True):
        """
        Define loss of TF graph
        :param y: correct labels
        :param model: output of the model
        :param mean: boolean indicating whether should return mean of loss
                     or vector of losses for each input of the batch
        :return: return mean of loss if True, otherwise return vector with per
                 sample loss
        """

        op = model.op
        if "softmax" in str(op).lower():
            logits, = op.inputs
        else:
            logits = model

        out = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)

        if mean:
            out = tf.reduce_mean(out)
        return out

    def _fgsm_k(self, model, fake_class_idx, epsilon, img, descent=False):
        fake_class_idx = fake_class_idx
        fake_target = to_categorical(fake_class_idx, 1000)
        fake_target_variable = K.variable(fake_target)
        loss = metrics.categorical_crossentropy(model.output, fake_target_variable)
        gradients = K.gradients(loss, model.input)
        get_grad_values = K.function([model.input], gradients)
        grad_values = get_grad_values([img])[0]
        grad_signs = grad_values.copy()
        grad_signs[grad_values < 0] = -1
        grad_signs[grad_values >= 0] = 1
        perturbation = grad_signs * epsilon

        adv = img + perturbation if descent is False else img - perturbation

        return adv, perturbation

    def _fgsm(self, x, predictions, eps, clip_min=None, clip_max=None, descent=False):
        """
        TensorFlow implementation of the Fast Gradient
        Sign method.
        :param x: the input placeholder
        :param predictions: the model's output tensor
        :param eps: the epsilon (input variation parameter)
        :param clip_min: optional parameter that can be used to set a minimum
                        value for components of the example returned
        :param clip_max: optional parameter that can be used to set a maximum
                        value for components of the example returned
        :return: a tensor for the adversarial example
        """
        # Compute loss
        y = tf.to_float(
            tf.equal(predictions, tf.reduce_max(predictions, 1, keep_dims=True)))

        loss = self.model_loss(y, predictions)
        # Define gradient of loss wrt input
        grad, = tf.gradients(loss, x)

        # Take sign of gradient
        signed_grad = tf.sign(grad)

        # Multiply by constant epsilon
        scaled_signed_grad = eps * signed_grad
        if descent:
            # Add perturbation to original example to obtain adversarial example
            adv_x = tf.stop_gradient(x - scaled_signed_grad)
        else:
            adv_x = tf.stop_gradient(x + scaled_signed_grad)

        # If clipping is needed, reset all values outside of [clip_min, clip_max]
        if (clip_min is not None) and (clip_max is not None):
            adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

        return adv_x

    def _fgsm_k_iter(self, model, fake_class_idx, epsilon, img, n_steps=10, descent=False):
        perturbation = None
        for ix in range(n_steps):
            if perturbation is not None:
                img = img + perturbation if descent == False else img - perturbation
            adv, perturbation = self._fgsm_k(model=model, fake_class_idx=fake_class_idx, epsilon=epsilon, img=img)

        perturbation = adv - img if descent is False else img - adv

        return adv, perturbation

    def fgsm(self, model, img, softmax_predictions, eps=0.9, descent=False):
        images_placeholder = tf.placeholder(tf.float32, shape=model.input_shape)
        predictions_placeholder = model(images_placeholder)
        adv_x = self._fgsm(images_placeholder, predictions_placeholder, eps=eps, descent=descent)
        with self.sess.as_default():
            feed_dict = dict()
            feed_dict[K.learning_phase()] = 0
            feed_dict[images_placeholder] = img
            feed_dict[predictions_placeholder] = softmax_predictions
            res = self.sess.run([adv_x], feed_dict=feed_dict)
        return res

    def fgsm_iter(self, model, img, softmax_predictions, n_steps=10, eps=0.9, descent=False):

        images_placeholder = tf.placeholder(tf.float32, shape=model.input_shape)
        predictions_placeholder = model(images_placeholder)
        adv_x = self._fgsm(images_placeholder, predictions_placeholder, eps=eps, descent=descent)
        with self.sess.as_default():
            for ix in range(n_steps):
                img_input = img if ix == 0 else res[0]
                feed_dict = dict()
                feed_dict[K.learning_phase()] = 0
                feed_dict[images_placeholder] = img_input
                feed_dict[predictions_placeholder] = softmax_predictions
                res = self.sess.run([adv_x], feed_dict=feed_dict)
        return res


