# CropRecApp
## Summary
CropRecApp is an android app thats uses a pre trained convolutional neural network to make image recognition of two different classes: Wheat and Maize. 

It also makes a recognition of the phones orientation using a pre trained artificial neural network with 8 different outcomes: North (N), Northeast (NE), Northwest (NW), East (E), South (S), Southeast (SE), Southwest (SW) and West (W).  

![crop_ic_launcher](https://user-images.githubusercontent.com/61889565/93372816-4c7ee480-f809-11ea-97f0-02983d4e496f.png)

## Pre requisites
An Android device or an emulator running API level 21 or higher. A minimum of 200 MB of internal storage space free is recommended.

## Use
There are two ways for the user to input the image to be tested by the model, the first is by taking the picture using the mobile phones camera and the second is by selecting it from the mobile phones internal memory.

![default_mainscreen_small](https://user-images.githubusercontent.com/61889565/93372812-4be64e00-f809-11ea-8b10-8865940e5eb1.png)

Once the image to be tested is selected, it is shown on screen and the app automatically makes the class prediction.

![testing_taken_image_small](https://user-images.githubusercontent.com/61889565/93372822-4db01180-f809-11ea-841c-0ebeef87f735.png)

The first time you run the app, it takes a little moment to make the first prediction, after that, the image classification runs immediately.

## Result

As a result, the image label is shown on screen inside a circle shown in the lower right side of the screen.

If the image result is to be a 'Wheat' image, then it appears a wheat icon inside the circle and its background is tinted on a yellow color.

![selected_wheat_output_small](https://user-images.githubusercontent.com/61889565/93372805-4852c700-f809-11ea-90e0-4dd2e0738258.png)

If the image result is to be a 'Maize' image, then it appears a maize icon inside the circle and its background is tinted on a green color.

![selected_maize_output_small](https://user-images.githubusercontent.com/61889565/93372808-4a1c8a80-f809-11ea-979b-302244441382.png)

In case the test image was taken using the mobile phones camera, its location and orientation is shown at the top of the screen.

![testing_taken_image_output_small](https://user-images.githubusercontent.com/61889565/93372820-4d177b00-f809-11ea-9ab4-fae32cae8819.png)

