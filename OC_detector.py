import cv2
import dlib
import numpy as np
from keras.models import load_model
from scipy.spatial import distance as dist
from imutils import face_utils
import time

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

def detect(img, cascade = face_cascade , minimumFeatureSize=(20, 20)):
    if cascade.empty():
        raise (Exception("There was a problem loading your Haar Cascade xml file."))
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=1, minSize=minimumFeatureSize)
    

    if len(rects) == 0:
        return []

    rects[:, 2:] += rects[:, :2]

    return rects

def cropEyes(frame):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	te = detect(gray, minimumFeatureSize=(80, 80))

	if len(te) == 0:
		return None
	elif len(te) > 1:
		face = te[0]
	elif len(te) == 1:
		[face] = te

	face_rect = dlib.rectangle(left = int(face[0]), top = int(face[1]),
								right = int(face[2]), bottom = int(face[3]))
	
	shape = predictor(gray, face_rect)
	shape = face_utils.shape_to_np(shape)

	(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
	(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

	leftEye = shape[lStart:lEnd]
	rightEye = shape[rStart:rEnd]

	l_uppery = min(leftEye[1:3,1])
	l_lowy = max(leftEye[4:,1])
	l_dify = abs(l_uppery - l_lowy)

	lw = (leftEye[3][0] - leftEye[0][0])

	minxl = (leftEye[0][0] - ((24-lw)/2))
	maxxl = (leftEye[3][0] + ((24-lw)/2)) 
	minyl = (l_uppery - ((24-l_dify)/2))
	maxyl = (l_lowy + ((24-l_dify)/2))
	
	left_eye_rect = np.rint([minxl, minyl, maxxl, maxyl])
	left_eye_rect = left_eye_rect.astype(int)
	left_eye_image = gray[(left_eye_rect[1]):left_eye_rect[3], (left_eye_rect[0]):left_eye_rect[2]]
	
	r_uppery = min(rightEye[1:3,1])
	r_lowy = max(rightEye[4:,1])
	r_dify = abs(r_uppery - r_lowy)
	rw = (rightEye[3][0] - rightEye[0][0])
	
	minxr = (rightEye[0][0]-((24-rw)/2))
	maxxr = (rightEye[3][0] + ((24-rw)/2))
	minyr = (r_uppery - ((24-r_dify)/2))
	maxyr = (r_lowy + ((24-r_dify)/2))
	
	right_eye_rect = np.rint([minxr, minyr, maxxr, maxyr])
	right_eye_rect = right_eye_rect.astype(int)
	right_eye_image = gray[right_eye_rect[1]:right_eye_rect[3], right_eye_rect[0]:right_eye_rect[2]]

	if 0 in left_eye_image.shape or 0 in right_eye_image.shape:
		return None
	left_eye_image = cv2.resize(left_eye_image, (24, 24))
	right_eye_image = cv2.resize(right_eye_image, (24, 24))
	right_eye_image = cv2.flip(right_eye_image, 1)
	
	return left_eye_image, right_eye_image 

def processData(img):
	img = img.astype('float32')
	img /= 255
	img = np.expand_dims(img, axis=2)
	img = np.expand_dims(img, axis=0)
	return img

def main():
	camera = cv2.VideoCapture(0)
	(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
	if int(major_ver)  < 3 :
		fps = camera.get(cv2.cv.CV_CAP_PROP_FPS)
		print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
	else :
		fps = camera.get(cv2.CAP_PROP_FPS)
		print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
	
	num_frames = 120
	print("Capturing {0} frames".format(num_frames))
	# Start time
    start = time.time()
	
	#print(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
	#print(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
	#print(camera.get(cv2.CAP_PROP_FPS))
	if not camera.isOpened():
		raise RuntimeError("Could not start camera.")
	
	model = load_model('model_aug1.hdf5')
	
	state = ''
	
	while True:
		
		ret, frame = camera.read()
		if ret:
			eyes = cropEyes(frame)
			if eyes is None:
				continue
			else:
				left_eye,right_eye = eyes
		
		prediction = (model.predict(processData(left_eye)) + model.predict(processData(right_eye)))/2.0
		print(prediction)
		#cv2.imwrite("right_eye.jpg",right_eye)
		
		if prediction[0][0] < 0.99 :
			state = 'open'
		else:
			state = 'closed'
		
		cv2.putText(frame, "State: {}".format(state), (300, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			
		cv2.imshow('blinks counter', frame)
		#cv2.imshow("right_eye.jpg",right_eye)
		#cv2.imshow("left_eye.jpg",left_eye)
		
		"""
		(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
		if int(major_ver)  < 3 :
			fps = camera.get(cv2.cv.CV_CAP_PROP_FPS)
			print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
		else :
			fps = camera.get(cv2.CAP_PROP_FPS)
			print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
		"""
		print(camera.get(cv2.CAP_PROP_FPS))
		cv2.waitKey(0)
		key = cv2.waitKey(1) & 0xFF

		if key == ord('q'):
			break
			
	end = time.time()
	seconds = end - start
	print("Time taken : {0} seconds".format(seconds))
	camera.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()