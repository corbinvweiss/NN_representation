# NN representation research
Recall that this MLP takes an image of a handwritten digit, such as an image of a 3, and outputs a vector of classification values such as `[0, 0, 0.01, 0.98, 0, 0, 0, 0.05, 0, 0]`. This is a classification vector, and the Neural Network simply classifies the image as the index of the highest value. But there is more information in this vector than just the highest value. It is informative to look at the distribution of other values, because the wrong values tell us which classes are closest to other in the network. 

To understand the internal representation of reality in a Neural Network I would like to average the classification vectors across all samples of each input class and look at the classification distributions.

# Analysis
The mean certainties of each classification were plotted for 1000 samples of each class. The results are consistent with thinking of classification certainty as a measure of distance, where the distance from class $a$ to class $b$ is inversely proportional to the certainty of classifying $a$ as an image of $b$. In particular, distances are nearly symmetric, and the distance from a class to itself is the smaller than any other distance. 

Using this analogy of classification certainty as distance, we used PCA to visualize the distances between classes. PCA accounted for only 26% of the variation in the certainties, but the distances between classes were consistent with the shapes of the images. For example, an 8 and a 3 are very similar in shape, and are very close in the PCA visualization. Similarly, a 0 and a 1 are quite different shapes, and are the most distant of the classes in the PCA graph. 

These results support the notion that the MLP learns to "see" digits just like humans do, and that the training process separates the different digits spatially in its internal representation of reality.

# Next Steps
Watch how the PCA visualization of the distances between classes in the MLP changes during the training process. To do this I should add a step between epochs which runs a few test samples through the MLP, calculates distances, runs PCA, and plots the results. Ideally this would be live-updating, allowing the user to step through the training process using the arrow keys, and storing the adjacency matrices as it goes along. At any point in the training cycle I want to be able to step back through the training process using the arrow keys and watch how the PCA viz has changed. 