import numpy as np
import cv2

cap = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
pepe = cv2.imread('pepe.png', -1)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
	    gray,
	    scaleFactor=1.1,
	    minNeighbors=5,
	    minSize=(30, 30),
	    flags = cv2.CASCADE_SCALE_IMAGE
	)
    #print "Found {0} faces!".format(len(faces))
    for (x, y, w, h) in faces:
		#cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

		pepeResize = cv2.resize(pepe, (h, w), interpolation = cv2.INTER_AREA)

		roi = frame[y:y+h, x:x+w]
		pepe2gray = cv2.cvtColor(pepeResize,cv2.COLOR_BGR2GRAY)
		mask = pepeResize[:,:,3]
		mask_inv = cv2.bitwise_not(mask)
		frameBG = cv2.bitwise_and(roi,roi,mask = mask_inv)
		pepeFG = cv2.bitwise_and(pepeResize,pepeResize,mask = mask)
		b,g,r,a = cv2.split(pepeFG)
		pepeFG = cv2.merge((b,g,r))
		dst = cv2.add(frameBG,pepeFG)
		frame[y:y+h, x:x+w] = dst

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

