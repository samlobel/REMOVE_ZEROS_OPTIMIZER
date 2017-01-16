# REMOVE ZERO GRADS
This helps train parameters with infrequent signal at a reasonable rate. There's a big difference between a really small gradient and zero gradient, because zero gradient means that the feature is totally unrelated to classification for a given sample. That means that by all rights, zeros shouldn't be factored into your gradient while taking the mean.

By looking at the error gradient w.r.t. each sample, and filtering out the zeros before taking the mean, you get stronger updates for sparse-but-important elements. 

It works roughly how you'd expect after that explanation. You get all the gradients, count how many samples have non-zero gradients for each feature, sum up the gradients, and divide by that number for all of the features (instead of by the total number of samples, like a true mean).

## RESULTS
I ran a few simple experiments, training a 4-layer NN using ADAM, Gradient Descent, Momentum, and my optimization procedure. My optimizer works better per-sample than SGD, a little better than momentum, and worse than ADAM. This is promising, because it would be easy to add momentum to my optimizer, to make it even better. Here are results, after different numbers of 32-sample minibatches.

It's clearly not a rigorous test, but I ran those experiments a lot of times, and the results are generally the same.
___


## MINE
| BATCH NUMBER        | ACCURACY           | CROSS ENTROPY  |
| ------------- |:-------------:| -----:|
| 0       | 0.1643999964      |   2.31621956825 |
| 500       | 0.919000029564      |   0.253862768412 |
| 1000      | 0.934499979019      |   0.22866782546 |
| 1500      | 0.948499977589      |   0.155483990908 |
| 2000      | 0.957300007343      |   0.145764634013 |


## SGD
| BATCH NUMBER        | ACCURACY           | CROSS ENTROPY  |
| ------------- |:-------------:| -----:|
| 0       | 0.101400002837      |   2.32768273354 |
| 500       | 0.907400012016      |   0.312989145517 |
| 1000      | 0.923500001431      |   0.254277586937 |
| 1500      | 0.940500020981     |   0.196873918176 |
| 2000      | 0.947700023651      |   0.164660334587 |


## MOMENTUM
| BATCH NUMBER        | ACCURACY           | CROSS ENTROPY  |
| ------------- |:-------------:| -----:|
| 0       | 0.101400002837      |   2.32119107246 |
| 500       | 0.910499989986      |   0.306798815727 |
| 1000      | 0.922100007534      |   0.25702804327 |
| 1500      | 0.939199984074     |   0.199875473976 |
| 2000      | 0.95349997282      |  0.149812713265 |




## ADAM
| BATCH NUMBER        | ACCURACY           | CROSS ENTROPY  |
| ------------- |:-------------:| -----:|
| 0       | 0.0931999981403      |   2.31224012375 |
| 500       | 0.933099985123      |   0.214252337813 |
| 1000      | 0.955200016499      |   0.144142642617|
| 1500      | 0.959699988365     |   0.133631378412 |
| 2000      | 0.962300002575      |  0.11844394356 |




