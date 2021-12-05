from FaceDetector import FaceDetector

if __name__ == "__main__":
    db_file = "src\\dataset\\db.csv"
    cascPath = 'src\\haarcascade_frontalface_default.xml'
    trainerPath = 'src\\trainer\\trainer.yml'
    frameCapture = 1000
    lower_th_ratio = 0.15 # 0.15 works relatively well here
    upper_th_ratio = 0.2 # 0.2 works relatively well here
    sigma = 1
    detector = FaceDetector(db_file, cascPath, trainerPath, frameCapture, sigma, lower_th_ratio, upper_th_ratio)

    print('Real Time Face Detection and Recognition')
    print('Project by Joshua Gracie, CAP 5415')

    detector.display_options()