#from tensorflow.keras.applications.vgg16 import VGG16
#from tensorflow.keras.preprocessing import image
#from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np


#from keras.engine import  Model
from keras.layers import Input
#from keras_vggface.vggface import VGGFace

from numpy import expand_dims
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
#from keras_vggface.utils import decode_predictions

#code from tutorial: https://machinelearningmastery.com/how-to-perform-face-recognition-with-vggface2-convolutional-neural-network-in-keras/
def extract_face(filename, required_size=(224, 224)):
	# load image from file
	pixels = pyplot.imread(filename)
	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array


def feature_extraction():
    file = open("data/names_with_numbers.txt",'r')
    names = []
    for line in file:
        line = line.strip().split()
        if int(line[1]) == 2:
            names.append(line[0])
        if len(names) == 251:
            print("Collected")
            break
        
    #model = VGG16(weights='imagenet', include_top=False)   
    input_shape = (224, 224, 3)
    model = VGGFace(model="resnet50",include_top=False, input_shape=input_shape, pooling='avg') # pooling: None, avg or max

    #model = VGG16(weights = 'imagenet', input_shape = (input_shape[0], input_shape[1], input_shape[2]), pooling = 'max', include_top = False)

    features_list = []
    counting = 0
    for name in names:
        filename1 = "data/lfw-deepfunneled/lfw-deepfunneled/lfw-deepfunneled/" + name + "/" + name + "_0001.jpg"
        filename2 = "data/lfw-deepfunneled/lfw-deepfunneled/lfw-deepfunneled/" + name + "/" + name + "_0002.jpg"
        filenames = [filename1,filename2]
        for image_path in filenames:
            #img = image.load_img(image_path, target_size=(224, 224))
            #img = image.load_img(image_path, target_size=(250, 250))
            #x = image.img_to_array(img)
            #x = np.expand_dims(x, axis=0)
            #x = preprocess_input(x)
            
        
            # load the photo and extract the face
            pixels = extract_face(image_path)
            # convert one face into samples
            pixels = pixels.astype('float32')
            samples = expand_dims(pixels, axis=0)
            # prepare the face for the model, e.g. center pixels
            samples = preprocess_input(samples, version=1)
            # create a vggface model
            #model = VGGFace(model='resnet50')
            # perform prediction
            yhat = model.predict(samples)
            
            features_list.append(yhat)
            counting+=1
            if counting % 20 == 0:
                print(counting)
            # convert prediction into names
            #results = decode_predictions(yhat)
            
            #features = model.predict(x)
            #features_list.append(features)
    print("Number of feature vectors:",len(features_list))
    print("Each feature vector is of length:",features_list[0].shape)
    #outfile = open("extractions/VGGFace_vgg_lfw-deepfunneled.txt",'w')
    outfile = open("extractions/VGGFace_resnet50_lfw-deepfunneled.txt",'w')
    for features in features_list:
        #print(np.linalg.norm(features))
        features = features/np.linalg.norm(features) #are we allowed to normalize?
        #print(np.linalg.norm(features))
        for i in range(features.shape[1]):
            outfile.write(str(features[0,i]))
            #print(i,str(features[0,i]))
            outfile.write(" ")
        outfile.write("\n")

    outfile.close()

if __name__ == "__main__":
    feature_extraction()
