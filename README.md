# Traffic

I started with a simple convolutional neural network, as the one on the lecture code, with the following structure:
* A convolutional layer. Learn 32 filters using a 3x3 kernel
* A max-pooling layer, using a 2x2 pool size
* A hidden layer with 128 units
* A dropout rate of 0.5 to prevent overfitting

This base model performed very poorly, with only 0.0548 accuracy and a high loss score.

With the goal of finding a model with the best testing accuracy, I then tested different modifications to the base model.

These are my experiment results:

|# | Modification | Testing accuracy|
|--|--------------|-----------------|
|1 | Base model   | 0.0548|
|2 | Add second convolutional layer, identical to the first | 0.9702|
|3 | Add second maxpooling layer (after the second convolutional layer), identical to the first | 0.9506|
|4 | Remove second maxpooling layer, increase kernal size in second convolutional layer to (4, 4) | 0.9448|
|5 | Double number of filters (to 64) in second convolutional layer | 0.9637|
|6 | Double number of filters (to 64) in the two convolutional layer | 0.9479|
|7 | Add second maxpooling layer (after the second convolutional layer), using a 3x3 pool size | 0.9195|
|8 | Change the pool size from 2x2 to 3x3 in the first maxpooling layer | 0.9550|
|9 | Add second hidden layer (both layers with 128 units) | 0.9383|
|10 | Decrease dropout rate to 0.3 | 0.9662|

Just by adding a second convolutional layer, the accuracy score improved to 0.9702 and none of the other tested models were able to top this accuracy score.


After these modifications, trying to maximise the accuracy score, my final convolutional neural network is made of:
* First convolutional layer. Learn 32 filters using a 3x3 kernel
* A max-pooling layer, using a 2x2 pool size
* Second convolutional layer. Learn 32 filters using a 3x3 kernel
* A hidden layer with 128 units
* A dropout rate of 0.5 to prevent overfitting
