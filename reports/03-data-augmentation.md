# Segmentation of cloud regions

## Data Augmentation

For the need of data augmentation let me just quote *Aurélien Géron* from the 2025 release of his book [Hands-On Machine Learning with Scikit-Learn and PyTorch](https://www.oreilly.com/library/view/hands-on-machine-learning/9798341607972/), p. 438:
> Data augmentation artificially increases the size of the training set by generating many realistic variants of each training instance. This reduces overfitting, making this a regularization technique. The generated instances should be as realistic as possible: ideally, given an image from the augmented training set, a human should not be able to tell whether it was augmented or not. Simply adding white noise will not help; the modifications should be learnable (white noise is not).
>
> For example, you can slightly shift, rotate, and resize every picture in the training set by various amounts and add the resulting pictures to the training set (...). To do this, you can use tools available in `torchvision.transforms.v2` (e.g., `RandomCrop`, `RandomRotation`, etc.). This forces the model to be more tolerant to variations in the position, orientation, and size of the objects in the pictures. You can similarly use transforms to tweak the colors, and contrasts to simulate many different lighting conditions. In general, you can also flip the pictures horizontally (except for text and other asymmetrical objects). By combining these transformations (using Compose), you can greatly increase your training set size.

So, basically we use it to
- get the model to be more tolerant to variations in the data
- increase our training set size

With this in mind we explored some (few) libraries and found the [albumentations](https://albumentations.ai/docs/) package quite promising and ready for the task. Using it, gives us these benefits:
- it is well prepared for image augmentation and also augments the masks, which in our case are the labels
- in opposite to the `imgaug` package, which we had examined at the beginning of the project, this one is actively maintained and runs with current Python versions

Setting it up is quite easy:

```python
import albumentations as A

pipeline = A.Compose([
    A.HorizontalFlip(p=0.3),
    A.VerticalFlip(p=0.3),
    A.RandomBrightnessContrast(p=0.8),
    A.Rotate(p=0.6)
])

# Call the pipeline giving it an image and the stack of its masks
# NOTE: `csi` is an instance of our custom CLoudSegmentationImage class which 
# gathers the labels (the masks)
transformed_data = pipeline(
    image=csi.get_image_rgb(),
    masks=csi.get_mask_stack()
)
```
