# Deconvnet
This is a implementation of Deconvnet in keras, following Matthew D.Zeiler's paper [Visualizing and Understanding Convolutional Networks](http://arxiv.org/pdf/1311.2901v3.pdf)

## Feature
Given a pre-trained keras model, this repo can visualize features of specified layer including dense layer.  

## Dependencies
* [Keras](https://github.com/fchollet/keras) 1.1
* Python >= 2.7
* argparse 1.0
* PIL 1.1

## Examples
Below is several examples of feature visualization based on pre-trained VGG16 in keras, 'max' means pick the greates activation in the feature map to be visualized and set other elements to zeros, 'all' mean use all values in the feature map to visualize.
* Original Image
  ![husky](https://raw.githubusercontent.com/Jallet/Deconvnet-keras/master/results/husky.jpg)
* block3_conv3_128
  * max <br />
  ![block3_conv3_128_max](https://raw.githubusercontent.com/Jallet/Deconvnet-keras/master/results/block3_conv3_128_max.png)
  * all <br />
  ![block3_conv3_128_all](https://raw.githubusercontent.com/Jallet/Deconvnet-keras/master/results/block3_conv3_128_all.png)
* block4_conv2_46
  * max <br />
  ![block4_conv2_46_max](https://raw.githubusercontent.com/Jallet/Deconvnet-keras/master/results/block4_conv2_46_max.png)
  * all <br />
  ![block4_conv2_46_all](https://raw.githubusercontent.com/Jallet/Deconvnet-keras/master/results/block4_conv2_46_all.png)
* block5_conv3_256
  * max <br />
  ![block5_conv3_256_max](https://raw.githubusercontent.com/Jallet/Deconvnet-keras/master/results/block5_conv3_256_max.png)
  * all <br />
  ![block5_conv3_256_all](https://raw.githubusercontent.com/Jallet/Deconvnet-keras/master/results/block5_conv3_256_all.png)
* fc1_0 <br />
![fc1_0_all](https://raw.githubusercontent.com/Jallet/Deconvnet-keras/master/results/fc1_0_all.png)
* fc2_248 <br />
![fc2_248_max](https://raw.githubusercontent.com/Jallet/Deconvnet-keras/master/results/fc2_248_max.png)
* predictions_248 <br />
predictions_248 is the predicted class of the image(label: Eskimo_dog)
![predictions_248_max](https://raw.githubusercontent.com/Jallet/Deconvnet-keras/master/results/predictions_248_max.png)

## Shortage
* The code implements visualize function for only Convolution2D, MaxPooling2D, Flatten, Input, Dense, Activation layers, thus cannot handle other type of layers.
* The code support only plain networks, thus cannot visualize ResNet, Highway Networks or something.
