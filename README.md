# Food-Recoginition

This code looks is building a convolutional neural network (CNN) using the Keras library. The CNN consists of a sequence of layers:

The first layer is a Convolution2D layer with 32 filters, a 3x3 kernel size, and a ReLU activation function. The input shape is specified as (64, 64, 3), which means the input data consists of 64x64 pixel images with 3 color channels (RGB).

The second layer is a MaxPooling2D layer with a pool size of (2, 2). This layer reduces the size of the feature maps produced by the Convolution2D layer by taking the maximum value in each 2x2 block.

The third layer is another Convolution2D layer with 32 filters and a ReLU activation function.

The fourth layer is another MaxPooling2D layer with a pool size of (2, 2).

The fifth layer is a Flatten layer, which flattens the output of the previous layer into a single long vector.

The sixth, seventh, eighth, ninth, and tenth layers are Dense layers with 32, 64, 128, 256, and 256 units respectively and ReLU activation functions. These are fully-connected (dense) layers that learn more complex patterns in the data.

The eleventh layer is a final Dense layer with 6 units and a softmax activation function. This layer outputs the class probabilities for each of the 6 classes.

The model is then compiled with the Adam optimization algorithm, categorical cross-entropy loss, and accuracy as metrics.

The training and test data are then loaded using the ImageDataGenerator class and the model is trained using the fit_generator method, which generates batches of data on the fly for training and validation. Finally, the trained model is saved to the file "model.h5".


Import the necessary Keras modules for building the CNN model:

Sequential: used to initialize the CNN model as a sequence of layers
Convolution2D: used to add a 2D convolutional layer to the CNN
Flatten: used to flatten the output of the convolutional layers into a single long vector
Dense: used to add a fully-connected (dense) layer to the CNN
MaxPooling2D: used to add a 2D pooling layer to the CNN
Initialize the CNN model as a Sequential model:

classifier = Sequential(): creates an instance of the Sequential class and assigns it to the classifier variable
Add the first convolutional and pooling layers to the CNN:

classifier.add(Convolution2D(32, (3,3), input_shape=(64,64,3), activation='relu')): adds a Convolution2D layer to the CNN with 32 filters, a 3x3 kernel size, and ReLU activation. The input shape of (64,64,3) specifies the size and number of channels of the input images.
classifier.add(MaxPooling2D(pool_size=(2,2))): adds a MaxPooling2D layer to the CNN with a pool size of (2,2). This layer reduces the size of the feature maps produced by the Convolution2D layer by taking the maximum value in each 2x2 block.
Add the second convolutional and pooling layers to the CNN:

classifier.add(Convolution2D(32, (3,3), activation='relu')): adds another Convolution2D layer to the CNN with 32 filters and a 3x3 kernel size.
classifier.add(MaxPooling2D(pool_size=(2,2))): adds another MaxPooling2D layer to the CNN with a pool size of (2,2).
Add the flattening layer to the CNN:

classifier.add(Flatten()): adds a Flatten layer to the CNN, which flattens the output of the previous layers into a single long vector.
Add the fully-connected (dense) layers to the CNN:

classifier.add(Dense(units=32, activation='relu')): adds a Dense layer to the CNN with 32 units and a ReLU activation function.
classifier.add(Dense(units=64, activation='relu')): adds another Dense layer to the CNN with 64 units and a ReLU activation function.
classifier.add(Dense(units=128, activation='relu')): adds another Dense layer to the CNN with 128 units and a ReLU activation function.
classifier.add(Dense(units=256, activation='relu')): adds another Dense layer to the CNN with 256 units and a ReLU activation function.
classifier.add(Dense(units=256, activation='relu')): adds another Dense layer to the CNN with 256 units and a ReLU activation
