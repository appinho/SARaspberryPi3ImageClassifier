# SqueezeNet
Deploy a pre-trained SqueezeNet Neural Network on a Raspberry Pi 3 with OpenCV 3.3.0

# Example
In Terminal redirect to SqueezeNet directory and try the command:

$ python squeeze_net_raspi.py --image test_images/drake.jpg

and get as a result following image:

![Screenshot](result_drake.png)

plus the print in the terminal of the top 5 estimated output classes with their probabilities:

[INFO] loading model...  
[INFO] classification took 0.4432 seconds  
[INFO] 1. label: drake, probability: 0.25705  
[INFO] 2. label: goose, probability: 0.18581  
[INFO] 3. label: black stork, probability: 0.10414  
[INFO] 4. label: hornbill, probability: 0.074497  
[INFO] 5. label: quail, probability: 0.051127  
