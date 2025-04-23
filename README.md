To train and deploy your own model:

Run final.py on your PC to begin training the neural network.

It will generate a trained model file named trained_module.pth.

Transfer trained_module.pth to your robot's Raspberry Pi (or equivalent setup).

On the robot, load the model in your control script to begin real-time image-based navigation.

Feel free to use as many epochs as you'd like—longer training can improve accuracy!

Make sure your dataset is properly labeled with directional arrows, and adjust the number of classes in the script if needed.

![FNN Guided Car](./guided_car.jpg)
