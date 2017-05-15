import _pickle as pickle
from adversarial_code.utils import Utils
from adversarial_code.cifar_keras_vgg import VGG
from adversarial_code.adversarial_tf import Adversarial

utils = Utils()
n_classes = 10
x_train, y_train, x_test, y_test = utils.load_cifar10(normalize=True)

adv_cls = Adversarial()
vgg = VGG(32, 32, 3)

model = vgg.model(dropout=True)
model.load_weights("normalized/unbalanced_4_vgg_custom.h5")

cur_class = 4

smp = utils.get_samples_by_class(x_test, y_test, cur_class, num_samples=1000)
epsilon = 0.01
file_name = str(epsilon)+"_unbalanced_class_"+str(cur_class)+"_array_adv.pickle"
print("Creating adversararies for: "+file_name)
adv = utils.get_adversaries(model, smp, cur_class, epsilon)
with open(file_name, 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(adv, f)