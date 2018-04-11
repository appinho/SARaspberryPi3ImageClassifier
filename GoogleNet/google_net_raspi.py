# Import packages
import numpy as np
import argparse
import time
import cv2

# Construct the argument parse
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# Define model and output labels
prototxt = "model/bvlc_googlenet.prototxt"
model = "model/bvlc_googlenet.caffemodel"
labels = "output_labels.txt"

# Load the class labels from disk
rows = open(labels).read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]
 
# Load the input image from disk
image = cv2.imread(args["image"])

# The CNN requires fixed spatial dimensions for the input image(s)
# so you need to ensure it is resized to 224x224 pixels while
# performing mean subtraction (104, 117, 123) to normalize the input

# after executing this command the "blob" now has the shape:
# (1, 3, 224, 224)
blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))

# Load the serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# Aet the blob as input to the network and perform a forward-pass to
# obtain the output classification
net.setInput(blob)
start = time.time()
preds = net.forward()
end = time.time()
print("[INFO] classification took {:.5} seconds".format(end - start))

# Sort the indexes of the probabilities in descending order (higher
# probabilitiy first) and grab the top-5 predictions
preds = preds.reshape((1, len(classes)))
idxs = np.argsort(preds[0])[::-1][:5]

# Loop over the top-5 predictions and display them
for (i, idx) in enumerate(idxs):
	# Draw the top prediction on the input image
	if i == 0:
		text = "Label: {}, {:.2f}%".format(classes[idx],
			preds[0][idx] * 100)
		cv2.putText(image, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX,
			0.7, (0, 0, 255), 2)

	# Display the predicted label
	# + associated probability to the console	
	print("[INFO] {}. label: {}, probability: {:.5}".format(i + 1,
		classes[idx], preds[0][idx]))

# Display the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
