# Imitation Learning Simulator

![alt_text](Sample.gif)

Full Video: [YouTube](https://www.youtube.com/watch?v=JJ3b7Vj7nx8&index=5&list=PLMr_u-BsTKSoWrumKl-4sDf_keQxDZFaa&t=7s)


Collected a dataset of images-steering angles on a regular beach track (consisted of forward-laps, reverse-laps, and recovery driving) as the track had a bias of left turns. This was augmented by using images from left and right cameras with biased steering angles.

The dataset was augmented by cropping (and normalizing) the image of the surrounding environment (only displays immediate road) and pruned to obtain a balanced dataset.

The CNN used was a slightly modified NVIDIA's end-to-end simulator with dropout and regularization. This was done to prevent overfitting from a 'rough' dataset and had tuned hyperparameters of learning rate, batch size, and early stopping patience/threshold.

Additionally, a moving average/low-pass filter is constructed in drive.py to smoothen out the driving behavior (and reduce the effect of any erratic steering angles) and passed to a PI controller.

<img src="Images/NVIDIA_CNN.png" width="500" height="600">



## Improvements
- Using a steering wheel to collect a more finely tuned steering angle dataset.
- Feed in a perspective transformed image (birds-eye view) in addition to the forward-facing view to the CNN to account for curvature/inclines in the track.
- Use transfer learning/ combine the architecture of another CNN (which takes in image input) at the flatten-layer.
