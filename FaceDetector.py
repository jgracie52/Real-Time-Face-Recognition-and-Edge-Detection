import cv2
import time
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import os
from CannyDetector import CannyDetector

class FaceDetector():
    def __init__(self, db_file, cascPath, trainerPath, frameCapture, sigma, lower_th_ratio, upper_th_ratio):
        """Initializes the Face Detector

        Args:
            db_file (string): the csv file containing name to id matches
            cascPath (string): path to the cascade xml file
            trainerPath (string): path to the recognition trainer
            frameCapture (int): number of frames to capture for new user faces
            sigma (double): sigma value to be used to generate gaussian masks for edge detection
            lower_th_ratio (double): value to use for the lower threshold ration for hysteresis thresholding
            upper_th_ratio (double): value to use for the upper threshold ration for hysteresis thresholding
        """

        self.db_file = db_file
        self.db = pd.read_csv(db_file)
        self.cascPath = cascPath
        self.trainer = trainerPath
        self.frame_capture = frameCapture
        self.canny = CannyDetector(sigma=sigma,lower_threshold_ratio=lower_th_ratio, upper_threshold_ratio=upper_th_ratio)

    def detect_faces(self):
        """Detects faces using OpenCV Cascade Classifier and a pretrained Haar Cascade Classifier for frontal face detection
        """

        print("\n [INFO] Detecting faces")

        # Setup classifier and video capture
        faceCascade = cv2.CascadeClassifier(self.cascPath)
        video_capture = cv2.VideoCapture(0)

        while True:
            # Capture frame-by-frame
            ret, frame = video_capture.read()

            # Set to gray scale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Get the faces
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Display the resulting frame
            cv2.imshow('Face Detector', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()

        print("\n [INFO] Exiting detector")
        self.display_options()

    def add_user(self, name):
        """Adds user to database and stores their name in store file

        Args:
            name (string): input name from the user
        """
        # Generate new id
        user_id = self.db['Id'].max() + 1
        print('[INFO] Adding user: {} as id: {}'.format(name, user_id))

        # Save the user in db file
        new_row = {'Id': user_id, 'Name': name}
        self.db = self.db.append(new_row, ignore_index=True)
        self.store_new_user()

        # Capture face
        self.capture_user_face(user_id)

        # Return to options
        self.display_options()
    
    def store_new_user(self):
        """Helper function to store the updated db_file
        """
        self.db.to_csv(self.db_file, index=False)

        
    def capture_user_face(self, user_id):
        """Captures a user's face given a generated user_id. Uses a Haar cascade classifier to detect 
        faces in the image and crop the resulting image to only include the detected faces

        Args:
            user_id (int): the generated id of the user
        """

        print('[INFO] Commencing face capture')
        print('[INFO] Please face webcam in...')

        # Stall for 5 seconds
        # Give user a chance to get ready
        for x in range(5, 0, -1):
            print('{}'.format(x))
            time.sleep(1)
        
        # Capture and store face
        video_capture = cv2.VideoCapture(0)
        faceCascade = cv2.CascadeClassifier(self.cascPath)

        # Create a new folder for the user
        Path("src\\dataset\\User." + str(user_id)).mkdir(parents=True, exist_ok=True)

        # Initialize individual sampling face count
        count = 0
        while(True):

            # Get the frame
            ret, img = video_capture.read()

            # Set to gray scale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = faceCascade.detectMultiScale( gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE)

            for (x,y,w,h) in faces:
                # Crop the image
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)     
                count += 1

                # Save the captured image into the datasets folder
                cv2.imwrite("src\\dataset\\User." + str(user_id) + '\\' +  
                            str(count) + ".jpg", gray[y:y+h,x:x+w])
                cv2.imshow('Capture', img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            elif count >= self.frame_capture: # Take x number of face samples as defined by frame_capture and stop video
                break

        # Do a bit of cleanup
        print("\n [INFO] Face has been successfully captured")
        video_capture.release()
        cv2.destroyAllWindows()

    def train_user_face(self):
        """Trains all user faces in the database file using the Local Binary Patterns Histograms (LBPH)
        classifier provided by OpenCV
        """

        # Path for face image database
        path = 'src\\dataset'
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        detector = cv2.CascadeClassifier(self.cascPath);

        print ("\n [INFO] Training faces. Please wait ...")
        # Function to get the images and label data
        faces,ids = self.get_images_and_labels(path, detector)

        # Train
        recognizer.train(faces, np.array(ids))

        # Save the model into trainer/trainer.yml
        recognizer.write(self.trainer) 

        # Print the numer of faces trained and end program
        print("\n [INFO] {0} faces trained successfully".format(len(np.unique(ids))))

        # Return to options
        self.display_options()

    def get_images_and_labels(self, path, detector):
        """Helper function to get the images and user names from the database file

        Args:
            path (string): the path to the dataset folder
            detector (object): Haar cascade classifier to get the faces from the images

        Returns:
            [type]: [description]
        """

        # Setup
        imagePaths = [f for f in Path('src/dataset').rglob('*.jpg')]     
        faceSamples=[]
        ids = []

        for imagePath in imagePaths:
            if imagePath == '':
                continue

            # Convert to gray scale
            PIL_img = Image.open(imagePath).convert('L') # grayscale
            img_numpy = np.array(PIL_img,'uint8')

            # Get the id of the user
            id = int(os.path.split(imagePath)[0].split('\\')[2].split('.')[1])

            # Get the faces
            faces = detector.detectMultiScale(img_numpy)

            for (x,y,w,h) in faces:
                # Append to samples
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)

        return faceSamples,ids

    def recognize_faces(self):
        """Recognizes faces using a trained LBPH classifier and a Haar cascade classifier for face localization
        """

        print("\n [INFO] Running recognizer")

        # Create and load LBPH classifier
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(self.trainer)

        # Create the Haar cascade classifier
        faceCascade = cv2.CascadeClassifier(self.cascPath);
        font = cv2.FONT_HERSHEY_SIMPLEX

        #iniciate id counter
        id = 0

        # names related to ids: example ==> Marcelo: id=1,  etc
        names = list(self.db['Name'])

        # Initialize and start realtime video capture
        cam = cv2.VideoCapture(0)
        cam.set(3, 640) # set video widht
        cam.set(4, 480) # set video height

        # Define min window size to be recognized as a face
        minW = 0.1*cam.get(3)
        minH = 0.1*cam.get(4)
        while True:
            # Get frame
            ret, img =cam.read()

            # Set to gray scale
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Get faces
            faces = faceCascade.detectMultiScale( gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30))

            for(x,y,w,h) in faces:
                # Crop face
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

                # Get the id of the user, if there is one
                id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
                
                # If confidence is less them 100 ==> "0" : perfect match 
                if (confidence < 100):
                    id = names[id-1]
                    confidence = "  {0}%".format(round(100 - confidence))
                else:
                    id = "unknown"
                    confidence = "  {0}%".format(round(100 - confidence))
                
                cv2.putText(
                            img, 
                            str(id), 
                            (x+5,y-5), 
                            font, 
                            1, 
                            (255,255,255), 
                            2
                        )
                cv2.putText(
                            img, 
                            str(confidence), 
                            (x+5,y+h-5), 
                            font, 
                            1, 
                            (255,255,0), 
                            1
                        )  
            
            cv2.imshow('Recognizer',img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Do a bit of cleanup
        print("\n [INFO] Exiting recognizer")
        cam.release()
        cv2.destroyAllWindows()
        self.display_options()

    def detect_edges(self):
        """Detects edges using Canny Edge Detection algorithm
        """

        print("\n [INFO] Detecting edges")

        # Setup
        video_capture = cv2.VideoCapture(0)

        while True:
            # Capture frame-by-frame
            ret, frame = video_capture.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #edges = cv2.Canny(gray, threshold1=100, threshold2=200)
            edges = self.canny.calculateEdges(gray)

            # Display the resulting frame
            cv2.imshow('Edge Detector', edges)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()

        print("\n [INFO] Exiting detector")
        self.display_options()

    def display_options(self):
        """Helper function to display program options
        """

        print('\nOptions:\n')
        print('1: Detect Faces (No recognition)')
        print('2: Add User to Recognizer')
        print('3: Retrain Faces')
        print('4: Recognize Faces')
        print('5: Detect Edges')
        print('6: Quit\n')

        option = input('')
        if option == '1':
            self.detect_faces()
        elif option == '2':
            name = input('Enter user name: ')
            self.add_user(name)
        elif option == '3':
            self.train_user_face()
        elif option == '4':
            self.recognize_faces()
        elif option == '5':
            self.detect_edges()
        elif option == '6':
            print('Exiting')
            return
        else:
            print('Not an option, please select from the given options')
            self.display_options()