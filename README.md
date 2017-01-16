# REMOVE ZERO GRADS
This helps train parameters with infrequent signal at a reasonable rate. There's a big difference between a really small gradient and zero gradient, because zero gradient means that the feature is totally unrelated to classification for a given sample. That means that by all rights, zeros shouldn't be factored into your gradient while taking the mean.

By looking at the error gradient w.r.t. each sample, and filtering out the zeros before taking the mean, you get stronger updates for sparse-but-important elements. 

It works roughly how you'd expect after that explanation. You get all the gradients, count how many samples have non-zero gradients for each feature, sum up the gradients, and divide by that number for all of the features (instead of by the total number of samples, like a true mean).

## RESULTS
I ran a few simple experiments, training a 4-layer NN using ADAM, Gradient Descent, Momentum, and my optimization procedure. My optimizer works better per-sample than SGD or momentum, and worse than ADAM. This is promising, because it would be easy to add momentum to my optimizer, to make it even better. Here are results, after different numbers of  32-sample minibatches.
___

