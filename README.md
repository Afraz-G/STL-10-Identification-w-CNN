# STL-10-Identification-w-CNN
A CNN built upon numpy that classifies an STL-10 dataset from http://cs.stanford.edu/~acoates/stl10 (created by Adam Coates, Honglak Lee, Andrew Y. Ng).
Dataset includes 10 classes (airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck) with each image being 96 by 96 pixels in color, consisting in pure binary data. 
Utilizes a maxpool of size 2, a Conv filter of size 3, and a softmax prediction process. 
This is a personal learning project and there may be instances of inefficiency and oversight and I will be trying my best to adjust and improve as well as apply this model for later projects if possible.

The model is not trained and currently this is a CNN with the hypothetical ability to determine (if i could determine the correct hyperparameters myself) an image of the aforementioned classes in RGB with 96x96 pixels. It utilizes CPU for computing as this is a learning opportunity for me to learn how CNNs work through numpy and i have attempted trial and error to determine the ideal hyperparameters but the training time simply takes too long. I will be leaving this as is for now and adjust/update/refine the model further in the near future as i would like to tackle other projects too.
