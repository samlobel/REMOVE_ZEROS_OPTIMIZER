# REMOVE ZERO GRADS
This is an idea I'm pretty sure will work. When taking the average gradient over samples, there are a lot of zero-gradients (especially in something like MNIST, which is sparse.) These really lower the average gradient, even though they signify that they are inconsequential, and should not be factored in to the average.

## Motivation
The original motivation for this was that the standard-deviation based learning wasn't working, because most parameters had a lot of samples have zero derivative. So there's this distribution, with a peak sticking out of it at zero. I thought about how to deal with this, and realized that filtering them out would help with regular optimization too. 


## What it helps with:
This helps train parameters with infrequent signal at a reasonable rate. Momentum actually slows down training these. If a parameter is activated every 100 samples, that means that on average it will train 100x slower than other parameters, even if it has a lot of representative power when it is active. In other words, it helps with sparse signals.

Furthermore, it will help with ReLus, because these by design zero out throughout their layers.
