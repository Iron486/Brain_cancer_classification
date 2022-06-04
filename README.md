# Brain_cancer_classification

The goal of the project was the classification of tumors based on MRI images.

There were 4 classes: 'glioma_tumor', 'meningioma_tumor', 'no_tumor' and 'pituitary_tumor'.

There were given the [train](https://github.com/Iron486/Brain_cancer_classification/tree/main/data/Training) and [test](https://github.com/Iron486/Brain_cancer_classification/tree/main/data/Training) datasets, both containing images belonging to the 4 classes. 
I fetched the data from here https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri.

In this repository there are 2 notebooks obtained with Jupyter Notebook and 2 Python scripts produced with Spyder :

- [EfficientNetB2.py](https://github.com/Iron486/Brain_cancer_classification/blob/main/EfficientNetB2.py)) that I used to fit an EfficientNetB2 model to the train dataset and predict on test dataset.
- [EfficientNetB6.py](https://github.com/Iron486/Brain_cancer_classification/blob/main/EfficientNetB6.py) in which I trained an EfficientNetB6 model and I predicted the model on test dataset.
- [EfficientNetB3.ipynb](https://github.com/Iron486/Brain_cancer_classification/blob/main/EfficientNetB3.ipynb) that is the model with the lowest accuracy. I applied 2 addictional hidden layers on top, and represented some images in the dataset and images with their respective predicted class. Moreover, I calculated some metrics (F1 score,precision,recall and accuracy) and I represented a confusion matrix.
- [CNN_with_convolutional_layers.ipynb](https://github.com/Iron486/Brain_cancer_classification/blob/main/CNN_with_convolutional_layers.ipynb) in which I fit a Convolutional Neural Network with an augmented train dataset and I predicted the model on test dataset. Furthermore, I represented the convolutional layers used to build the model visualizing the application of convolutional filters to a randomly picked image.


Below, I reported the training curves represented for the [notebook](https://github.com/Iron486/Brain_cancer_classification/blob/main/EfficientNetB3.ipynb) with the highest accuracy and lowest loss.

![EfficientNetB3](<p align="center"> <img src="https://user-images.githubusercontent.com/62444785/172028786-b25919f7-a963-4e51-8fb3-53a04633ce47.png" width="570" height="320"/>   </p>)
