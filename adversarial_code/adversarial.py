import caffe
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tempfile

class Adversarial(object):

    def __init__(self):
        self.caffe_root = '/Users/rafaelpossas/Dev/caffe'
        self.imagenet_labels_filename = self.caffe_root + '/data/ilsvrc12/synset_words.txt'
        self.labels = np.loadtxt(self.imagenet_labels_filename, str, delimiter='\t')
        self.prototxt = '/Users/rafaelpossas/Dev/caffe/models/bvlc_googlenet/deploy.prototxt'
        self.caffe_model = '/Users/rafaelpossas/Dev/caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel'

        caffe.set_mode_cpu()
        self.net = self.load_model()

        #pd.set_option('display.mpl_style', 'default')

    def pre_process(self,img_path):
        return self.transformer.preprocess('data',caffe.io.load_image(img_path))

    def load_model(self):
        BATCH_SIZE = 1
        net = caffe.Net(self.prototxt, self.caffe_model, caffe.TEST)
        # change batch size to 1 for faster processing
        # this just means that we're only processing one image at a time instead of like 50
        shape = list(net.blobs['data'].data.shape)
        shape[0] = BATCH_SIZE
        net.blobs['data'].reshape(*shape)
        net.blobs['prob'].reshape(BATCH_SIZE, )
        net.reshape()

        # Caffe comes with a handy transformer pipeline so that
        # we can make our images into the format it needs!
        self.transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_raw_scale('data',255)  # the reference model operates on images in [0,255] range instead of [0,1]
        self.transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB

        return net

    def display(self,data):
        plt.imshow(self.transformer.deprocess('data', data))

    def get_label_name(self,num):
        options = self.labels[num].split(',')
        # remove the tag
        options[0] = ' '.join(options[0].split(' ')[1:])
        return ','.join(options[:2])

    def predict(self,data, n_preds=6, display_output=True):
        self.net.blobs['data'].data[...] = data
        if display_output:
            self.display(data)
        prob = self.net.forward()['prob']
        probs = prob[0]
        top_k = probs.argsort()[::-1]
        for pred in top_k[:n_preds]:
            percent = round(probs[pred] * 100, 2)
            # display it compactly if we're displaying more than the top prediction
            pred_formatted = "%03d" % pred
            if n_preds == 1:
                format_string = "label: {cls} ({label})\ncertainty: {certainty}%"
            else:
                format_string = "label: {cls} ({label}), certainty: {certainty}%"
            if display_output:
                print format_string.format(
                    cls=pred_formatted, label=self.get_label_name(pred), certainty=percent)
        return prob

    def compute_gradient(self, image, intended_outcome):
        self.predict(image, display_output=False)
        # Get an empty set of probabilities
        probs = np.zeros_like(self.net.blobs['prob'].data)
        # Set the probability for our intended outcome to 1
        probs[0][intended_outcome] = 1
        # Do backpropagation to calculate the gradient for that outcome
        gradient = self.net.backward(prob=probs)
        return gradient['data'].copy()[0]

    def fast_gradient(self,img, step_size, gradient,inverse=False):
        delta = np.sign(gradient)
        if not inverse:
            return img + (step_size * delta)
        else:
            return img - (step_size * delta)

    def iterative_fast_gradient(self, img, img_label, step_size=0.1, n_steps=10):
        # maintain a list of outputs at each prediction
        prediction_steps = []
        grad = []
        for _ in range(n_steps):
            grad = self.compute_gradient(img, img_label)
            img = self.fast_gradient(img, step_size, grad)
            preds = self.predict(img, display_output=False)
            prediction_steps.append(np.copy(preds))
        print grad
        return (img, prediction_steps)

    def trick(self, image, desired_output, n_steps=1, step_size=0.1):
        # maintain a list of outputs at each step
        return self.iterative_fast_gradient(image, desired_output, step_size, n_steps)

    def make_less_like(self, image, label):
        _ = self.predict(image, display_output=False)
        grad = self.compute_gradient(image, label)
        resulting_img = self.predict(self.fast_gradient(image, 0.9, grad, inverse=True))
        return resulting_img

    def plot_steps(self, steps, label_list, **args):
        d = {}
        for label in label_list:
            d[self.get_label_name(label)] = [s[0][label] for s in steps]
        df = pd.DataFrame(d)
        df.plot(**args)
        plt.xlabel('Step number')
        plt.ylabel('Probability of label')