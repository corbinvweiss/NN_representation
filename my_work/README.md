# NN representation research
Recall that this MLP takes an image of a handwritten digit, such as an image of a 3, and outputs a vector of classification values such as `[0, 0, 0.01, 0.98, 0, 0, 0, 0.05, 0, 0]`. This is a classification vector, and the Neural Network simply classifies the image as the index of the highest value. But there is more information in this vector than just the highest value. It is informative to look at the distribution of other values, because the wrong values tell us which classes are closest to other in the network. 

To understand the internal representation of reality in a Neural Network I would like to average the classification vectors across all samples of each input class and look at the classification distributions.

