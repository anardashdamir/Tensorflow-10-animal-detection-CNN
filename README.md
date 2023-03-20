# Animal Classification using EfficientNetB0
This TensorFlow model predicts 10 classes of animals - butterfly, cat, chicken, cow, dog, elephant, horse, ragno, sheep, and squirrel, using transfer learning with EfficientNetB0 architecture. The model has been trained on a dataset of animal photos with corresponding labels.

**Requirements**
* TensorFlow 2.x
* numpy
* matplotlib
* pillow

**Dataset**

**Kaggle**: https://www.kaggle.com/datasets/alessiocorrado99/animals10

The dataset used to train the model is not included in this repository.The photos should be organized in folders named after the corresponding class. For example, all photos of cats should be in a folder named "cat", all photos of dogs should be in a folder named "dog", and so on. Otherwise, just use **preprocessing.ipynb**

**Results**

The model achieves 97% accuracy on the test set after 5 epochs of training.

