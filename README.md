# Fashion-MNIST-Ai

Build and train a convolutional neural network (CNN) model to recognize and classify images of clothing using the Fashion-MNIST dataset.

Here's an example of how the data looks:

![](https://github.com/EliFebres/Fashion-MNIST-Ai/blob/staging/images/fashion-mnist-logo.png)

## Model

I copied the TinyVGG, used in the [CNN explainer](https://poloclub.github.io/cnn-explainer/), to help construct the model. The provided link goes in depth into CNNs and the TinyVGG model, so there is no need to explain it here.

Model Stats:
Train loss: 0.22176 | Train accuracy: 91.83% | Test loss: 0.27062 | Test accuracy: 90.36%

## Model Evaluation
I believe the model's performance is more accurate than its raw accuracy percentage suggests. The low quality 28x28 images can sometimes lead to misclassifications, even by a human observer. If we were to exclude the images that humans would struggle to distinguish, it should improve the model's accuracy and better represent its true capabilities.

For instance, in the 3x3 grid of incorrect predictions, we see some challenges. On the last row, the second and third image could easily be misclassified by humans, let alone a simple TinyVGG model.

![](https://github.com/EliFebres/Fashion-MNIST-Ai/blob/staging/images/3x3-grid-of-missclassifications.png)

Despite these data challenges, I consider the model successful as it correctly and consistently classifies images that one would expect a model of this nature to get correct.

## License
This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <https://unlicense.org>
