import _pickle as pickle
import numpy as np

nb_classes = 10
epsilon = 0.01
root_folder = "../adversarial_code/data/"
x_all_adv = None
y_all_adv = None

for cls in range(nb_classes):
    file_name = str(epsilon) + "_class_" + str(cls) + "_array_adv.pickle"
    with open(root_folder+file_name, 'rb') as f:
        x_adv = pickle.load(f)
        x_all_adv = np.vstack((x_all_adv, x_adv)) if not x_all_adv is None else x_adv
        y_all_adv = np.vstack((y_all_adv, np.full((1000, 10), np.eye(10)[cls])))if not y_all_adv is None else np.full((1000, 10), np.eye(10)[cls])

with open("0.01_all_adv.pickle", 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    adv_arr = [x_all_adv, y_all_adv]
    pickle.dump(adv_arr, f)