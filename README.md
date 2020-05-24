# Face mask detector

The project's goal is to detect person wearing mask on his/her face or not.

## Steps taken to build this project

+ Train a custom DL model to detect whether person is or is not wearing a mask
+ Use this custom DL model to detect person/s with face mask or not in given image or real-time video stream

## Dataset to train a custom DL model

The dataset was created by [Prajna Bhandary](https://www.linkedin.com/feed/update/urn%3Ali%3Aactivity%3A6655711815361761280/)
This dataset consists of 1,376 images belonging to 2 classes:

+ `with_mask` : 690 images
+ `without_mask` : 686 images

To create this dataset, Prajna had the ingenious solution of:

+ Taking normal images of faces
+ Then creating a computer vision Python script to add face masks to them, thereby creating an artificial (but still real-world applicable) dataset.

To know more about creating this dataset, I would suggest:

+ Refer to [Prajna's Github repository](https://github.com/prajnasb/observations/tree/master/mask_classifier/Data_Generator)
+ [PyImageSearch blog](https://www.pyimagesearch.com/2018/11/05/creating-gifs-with-opencv/)
