# GoogleNet
Deploy a pre-trained GoogleNet Neural Network on a Raspberry Pi 3 with OpenCV 3.3.0

# Example
In Terminal redirect to GoogleNet directory and try the command:

$ python google_net_raspi.py --image test_images/space_shuttle.jpg

and get as a result following image:

![Screenshot](result_space_shuttle.png)

plus the print in the terminal of the top 5 estimated output classes with their probabilities:

[INFO] loading model...  
[INFO] classification took 0.85005 seconds  
[INFO] 1. label: space shuttle, probability: 0.81594  
[INFO] 2. label: missile, probability: 0.026718  
[INFO] 3. label: hair slide, probability: 0.025129  
[INFO] 4. label: projectile, probability: 0.02368  
[INFO] 5. label: warplane, probability: 0.0232  
