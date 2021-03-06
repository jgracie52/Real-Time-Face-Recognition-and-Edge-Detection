# Real time Face Recognition and Edge Detection
In this project, we implement a real time face detection and recognition system
as well as a near real time canny edge detector

### Face detection
Face detection is done using a pretrained Haar cascade classifier provided by OpenCV and trained to 
recognize frontal faces

### Face Recognition
Face recognition is achieved using a Local Binary Patterns Histograms (LBPH) classifier using images
capture from the system and cropped using the face detector

### Edge Detection
Edge detection is done using the Canny Edge Detector algorithm and is implemented in near real time (1 frame per 1.5 seconds approximately)

### How to Use
To use the program, first open main.py and adjust settings as desired. From there, run main.py and read
given options from the system:


Options:

1: Detect Faces (No recognition)
2: Add User to Recognizer
3: Retrain Faces
4: Recognize Faces
5: Detect Edges
6: Quit


Select option 1 to just detect faces without reconition
Select option 2 to add a user's face to the reconizer
Select option 3 to retrain all faces in db.csv
Select Option 4 to perform recognition
Select option 5 for edge detection
Select option 6 to quit

---
**NOTE**

Press q during any video capture to end the capture and return to the options menu

---
---
**NOTE**

A user may already be in the database file with the same face. If so, remove them from db.csv and delete their dataset folder. Then run option 2

---
---
**NOTE**

If there are no users in db.csv, or the trainer.yml file is missing, this will not work

---