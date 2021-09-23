from keras.models import load_model
import pickle
import cv2


def predict(img):
	image = cv2.imread(img)
	image = cv2.resize(image, (16, 37))
	# cv2.imshow('image', image)
	cv2.waitKey(0)
	image = image.astype("float") / 255.0


	image = image.flatten()
	image = image.reshape((1, image.shape[0]))

	model = load_model('./models/simple_nn')
	lb = pickle.loads(open('./models/simple_nn.pickle', "rb").read())

	preds = model.predict(image)

	i = preds.argmax(axis=1)[0]
	label = lb.classes_[i]

	text = "{}: {:.2f}%".format(label, preds[0][i] * 100)

	return text
