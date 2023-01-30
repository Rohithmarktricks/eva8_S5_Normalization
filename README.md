## EVA-8 Assignment 5

### Problem Statement

1. You are making 3 versions of your 4th assignment's best model (or pick one from best assignments):
Network with Group Normalization
Network with Layer Normalization
Network with L1 + BN

#### You MUST:
1. Write a single model.py file that includes GN/LN/BN and takes an argument to decide which normalization to include
2. Write a single notebook file to run all the 3 models above for 20 epochs each
Create these graphs:
    Graph 1: Test/Validation Loss for all 3 models together
    Graph 2: Test/Validation Accuracy for 3 models together
    graphs must have proper annotation
3. Find 10 misclassified images for each of the 3 models, and show them as a 5x2 image matrix in 3 separately annotated images. 
4. write an explanatory README file that explains:
    what is your code all about,
    how to perform the 3 normalizations techniques that we covered(cannot use values from the excel sheet shared)
    your findings for normalization techniques,
    add all your graphs
    your 3 collection-of-misclassified-images 



### Solution
#### Please refer to [Solution.ipynb](/Solution.ipynb) Jupyter Notebook for the experiments and plots.


#### Modules-API developed for future purposes!
#### [model.py](/model.py)
1. This is the main module that contains the code for the Model Definition.
2. This module contains the 3 classes:
    LayerNormCNN
    GroupNormCNN
    BatchNormCNN
3. This module also contains ```get_model()``` function that takes the ```normalization``` argument and creates the CNN model with corresponding Normalization technique.


#### [train_helper.py](/train_helper.py)
1. This is the helper module that contains the following:
    1. ```BN_Trainer:``` This class contains the train and test methods that have been written
    exclusively for the ```Batch Normalization + L1 Regulization```.
    2. ```Trainer:``` This class contains the train and test methods that can be used for 
    ```LayerNormalization``` and ```GroupNormalization```.
    3. Each of the above classes also contain ```get_stats()``` method, to track the loss and accuracy values.
2. ```get_misclassified_images```: this method takes the model and test_loader and generates the mis-classified images.


#### Loss of 3 different Normalization Techniques
![alt text](/images/Loss_plot.png)

#### Accuracy of CNN models with 3 different Normalization Techniques
![alt text](/images/Acc_plot.png)

#### Sample Misclassifications by the CNN model - Batch Normalization
![alt text](/images/BatchNorm_misclassified.png)

#### Sample Misclassifications by the CNN model - Layer Normalization
![alt text](/images/LayerNorm_misclassified.png)

#### Sample Misclassification by the CNN model - Group Normalization
![alt text](/images/GroupNorm_misclassified.png)


#### Findings:
1. BatchNormalization + L1 Regularization, seems to have greater convergence among other normalization techniques. But the loss function is not completely smooth.
2. Group Normalization: After having trained for 20 epochs (3-4 times), the common observation is that the starting error(rate) is minimum. Seems bit unstable (both loss and accuracy, seems to raise/fall rapidly)
3. Layer Normalization: Has the smooth loss curve among the three. Also, convergence rate is higher.
