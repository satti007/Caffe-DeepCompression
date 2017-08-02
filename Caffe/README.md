# Caffe
**Project Details:**
- Classified ImageNet(cats, dogs) dataset by **CaffeNet** and achieved an accuracy of **43.23%**
- Retrained an existing model( **CaffeNet** ) to perform a classification on ImageNet(cats, dogs) dataset and achieved an accuracy of **94.23%** \*
- Built a basic CNN model( **BasicNet** ) from scratch to perform a classification on ImageNet(cats, dogs) dataset and achieved an accuracy of **88.9%** \*   (\*limited the models to 10k iterations for comparison)

**Cats and Dogs dataset:**
- The dataset is taken from kaggle.As their test images don&#39;t have labels we divided the given training data set(25K) itself into the train(22K), validation(2K), test datasets(3K) with an equal composition of cats and dogs.

**1. Caffenet**
- We classified the data set using **caffenet** which was originally built to classify Imagenet data(1000 categories).This model doesn't just tell us whether an image is a cat/dog but rather gives us the name of the breed of cat/dog(as different breeds of animals are grouped into different categories in Imagenet).Here we used only top-1 predictions of the model and calculated the accuracy by considering that if the prediction of an animal(image) by this model is different i.e other than any breed of that animal, then we took that the model predicted it as another animal.We explain this by the following example, let us consider we have an image of cat and model predicted it as X.If X doesn't belong to set containing all breeds(breeds present in Imagenet categories) of a cat then we take that the model predicted it as a dog.
<p align="center"> 
<img src="https://github.com/satti007/Caffe-DeepCompression/blob/master/Caffe/Con_matrices/caffenet_con.png">
</p>
<p align="center"> 
Fig 1:Confusion matrix using caffenet 
</p>

- The accuracy **(43.13%)** here we got is in the range of caffenet top-1 accuracy for the Imagenet dataset **(57.4%)**
- Most of the images are misclassified as some other animal due to their similarity with those animals
<p align="center">
  <img src="https://github.com/satti007/Caffe-DeepCompression/blob/master/Caffe/Con_matrices/a1.jpg" height="100" width="100" /> ------->
  <img src="https://github.com/satti007/Caffe-DeepCompression/blob/master/Caffe/Con_matrices/a2.jpg" height="100" width="100" />  
  <img src="https://github.com/satti007/Caffe-DeepCompression/blob/master/Caffe/Con_matrices/A1.jpg" height="100" width="100" /> ------->
  <img src="https://github.com/satti007/Caffe-DeepCompression/blob/master/Caffe/Con_matrices/A2.jpg" height="100" width="100" />
</p>

**2. Retrained caffenet through transfer learning**
- To avoid the misclassification scenarios(like above),we retrain the model using only cats and dogs data through transfer learning.As we can observe the accuracy\* (after 10K iterations)of this model is way higher than that of the previous model.
<p align="center"> 
<img src="https://github.com/satti007/Caffe-DeepCompression/blob/master/Caffe/Con_matrices/caffenet_retrain.png">
</p>
<p align="center"> 
Fig 2: Confusion matrix using caffenet after ReTraining
</p>

**3. Basicnet(CNN built from scratch)**

<p align="center"> 
<img src="https://github.com/satti007/Caffe-DeepCompression/blob/master/Caffe/Con_matrices/basicnet_con.png">
</p>
<p align="center"> 
Fig 3: Confusion matrix using basicnet
</p>

- The above  model is the one we built , the accuracy\* (after 10K iterations)is not high as the caffenet retrained model because we observed that the training loss is not converging to zero(it&#39;s stagnant at 0.3 ) due to lack of sufficient training data and also we are using a kernel of size 5 with stride 2 in the first convolution layer where a lot of info of image is lost.One can argue that caffenet is using the kernel of size 11 with stride 4(in a conv1 layer) but their model is getting good accuracy, that&#39;s because of caffenet is deeper than our model and also uses norm layers to reconstruct the lost distribution of the data.So in order to improve our model accuracy we have decided to us only kernels of size 3 with stride 1 in our model and also made it deeper(added one more conv layer).We also increased our training dataset by **10X** using data augmentation techniques followed in Alexnet training and added dropout layers to avoid overfitting The results(accuracy\*)(after 10K iterations) are as follows,

<p align="center"> 
<img src="https://github.com/satti007/Caffe-DeepCompression/blob/master/Caffe/Con_matrices/basicnet_aug_con.png">
</p>
<p align="center"> 
Fig 4: Confusion matrix using basicnet(on augmented dataset)
</p>

- If we have left the BasicNet model for more iterations for training then it might have achieved a higher accuracy  than the re-trained CaffeNet model.

\*Accuracy-- limited the models to 10K iterations for comparison
