from score_detector import ScoreDetector
import cv2
detector = ScoreDetector()

screen = cv2.imread("score_screen_error.png")

print(detector.predict(screen))

screen = cv2.imread("score_screen_right.jpg")

print(detector.predict(screen))

screen = cv2.imread("right.jpg")

print(detector.predict(screen))