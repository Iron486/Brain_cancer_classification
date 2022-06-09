# Brain_cancer_classification

The goal of the project was the classification of tumors based on MRI images.

There were 4 classes: ` 'glioma_tumor', 'meningioma_tumor', 'no_tumor' and 'pituitary_tumor' `.

There were given the [train](https://github.com/Iron486/Brain_cancer_classification/tree/main/data/Training) and [test](https://github.com/Iron486/Brain_cancer_classification/tree/main/data/Training) datasets, both containing images belonging to the 4 classes. 
I fetched the data from here https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri.

In this repository there are 2 notebooks obtained with Jupyter Notebook and 2 Python scripts produced with Spyder :

- [EfficientNetB2.py](https://github.com/Iron486/Brain_cancer_classification/blob/main/EfficientNetB2.py)) that I used to fit an `EfficientNetB2` model to the train dataset and predict on test dataset.
- [EfficientNetB6.py](https://github.com/Iron486/Brain_cancer_classification/blob/main/EfficientNetB6.py) in which I trained an `EfficientNetB6` model and I predicted the model on test dataset.
- [EfficientNetB3.ipynb](https://github.com/Iron486/Brain_cancer_classification/blob/main/EfficientNetB3.ipynb) that is the model with the lowest accuracy. I applied 2 addictional hidden layers on top of a pre-trained `EfficientNetB6` model , and represented some images in the dataset and images with their respective predicted class. Moreover, I calculated some metrics (F1 score,precision,recall and accuracy) and I represented a confusion matrix.
- [CNN_with_convolutional_layers.ipynb](https://github.com/Iron486/Brain_cancer_classification/blob/main/CNN_with_convolutional_layers.ipynb) in which I fit a **Convolutional Neural Network** with an augmented train dataset and I predicted the model on test dataset. Furthermore, I represented the convolutional layers used to build the model visualizing the application of convolutional filters to a randomly picked image.

There are also the folder [plots](https://github.com/Iron486/Brain_cancer_classification/tree/main/plots) containing all the saved plots, [data](https://github.com/Iron486/Brain_cancer_classification/tree/main/data) that contains the train and test datasets and [models](https://github.com/Iron486/Brain_cancer_classification/tree/main/models) in which are the saved models (unfortunately, I could only load one model because the size of the saved models is way too big for the repository).
Below, I reported the training curves represented for the [notebook](https://github.com/Iron486/Brain_cancer_classification/blob/main/EfficientNetB3.ipynb) with the highest accuracy and lowest loss.

<p align="center"> <img src="https://user-images.githubusercontent.com/62444785/172028786-b25919f7-a963-4e51-8fb3-53a04633ce47.png" width="610" height="430"/>   </p>

**The model reached a 98.16% accuracy with a loss of 0.055**.

Here is a table with other metrics:

&nbsp;

|precision    |recall | f1-score  | support |
|----------------|---------|--------|---------|
|    glioma_tumor    |   0.99  |    0.97  |    0.98    |   139 | 
|meningioma_tumor     |  0.97  |    0.99  |    0.98  |     141|
  |      no_tumor     |  0.99   |   0.99   |   0.99   |     75|      
| pituitary_tumor     |  0.99    |  0.99  |    0.99 |     135|

&nbsp; 

An important metric is the `precision` (ratio between true positive and true positive plus false positive) calculated for the `no_tumor` class.

It's crucial because when the number of **false positive** (people that have a tumor, but the prediction belongs to `no_tumor` class) is high, it means that a lot of people with a tumor don't have a tumor according to the model. 

In this case, only one image out of 75 is a false positive as we can see in the confusion matrix represented below using `Seaborn` and `Matplotlib`.

<p align="center"> <img src="https://github.com/Iron486/Brain_cancer_classification/blob/main/plots/EfficientNetB3_confusion_matrix.png" width="735" height="590"/>   </p>

Moreover, I plotted some images in the datasets and the respective predicted class.

&nbsp;

![80ValidationDatasetimages_and_predictedclass_white](https://user-images.githubusercontent.com/62444785/172029617-9e20e656-57b6-4195-a1ea-8cee283c3392.png)


Below, instead, I plotted the first, an intermediate and the last convolutional layers related to [this notebook](https://github.com/Iron486/Brain_cancer_classification/blob/main/CNN_with_convolutional_layers.ipynb).

&nbsp;

<p align="center"> <img src="https://user-images.githubusercontent.com/62444785/172029736-08fa0703-4807-4a67-9c9e-b273e1e81e20.png" width="2435" height="80"/> </p>




<p align="center"> <img src="https://user-images.githubusercontent.com/62444785/172029739-9318d29a-0d92-46f6-a9f9-169950128c7f.png" width="2435" height="23"/> </p>




<p align="center"> <img src="https://user-images.githubusercontent.com/62444785/172029746-67c42131-0aea-460f-a3ea-e6b8e7013869.png" width="2435" height="9"/> </p>

&nbsp;

Clicking on the images, it can be noticed that the deeper we go, the less specific are the filters and the image is reduced in size, too.
