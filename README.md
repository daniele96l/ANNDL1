# Face-Recognition

Dataset: https://drive.google.com/file/d/18OqVM14td3Uvq-dXTE8Lpk9StL6lK_x1/view?usp=sharing

Task: classify images according to
All the people in the image are wearing a mask
No person in the image is wearing a mask
Someone in the image is not wearing a mask

For our work, we had to implement and train a convolutional neural network to perform an effective classification of pictures. In our case our dataset there were roughly 5000 pictures divided into 3 classes, people with masks, no masks, and some people with and without.
We choose to use a network that used convolution since it drastically improves the performance of the model.
We wrote and trained two types of models, one with transfer learning technique and one without.
For both cases due to the scarcity of images in the dataset, data augmenting was implemented.
For the model without we tried two approaches, one with sequential model and one using parallel levels of convolution, in a similar way that the GoogLeNet works. In both cases, we obtained scores between 0.6 and 0.7. For this reason, we decided to try transfer learning.
For the model with transfer learning, we did many tests, with different famous neural networks like Resnet50, 150, VGG, GoogLeNet, Xception, but in the end, we settled with the one that seemed more promising, DenseNet169.
In the beginning, we noticed that it was able to obtain fairly good results, but it seemed to overfit, for this reason, we added 3 dropout layers that had reduced overfitting and improved the overall performances.
We tried not only to freeze the first layers of the net, but also to train back the whole net from 0, and is here that we reached the best result, where the test accuracy on Kaggle was over 93%. This, also thanks to the technique of adaptive learning rate and early stopping.
More tests were done, with freezing more or fewer layers of the net, with weight decay and not, but in the end, the former model was the best performing.
