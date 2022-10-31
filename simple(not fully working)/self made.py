import cv2
import mediapipe as mp


def capture(input=0):
    cap = cv2.VideoCapture(input)
    if not cap.isOpened():
        return False
    else:
        return cap


mpHands = mp.solutions.hands
mpDrawing = mp.solutions.drawing_utils
hands = mpHands.Hands()
fingers_coord = [(8, 6), (12, 10), (16, 14), (20, 18)]
thumb_coord = [(4, 2)]

cap = capture()
while cap.isOpened():
    ret, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image)
    multi_landmarks = result.multi_hand_landmarks

    if multi_landmarks:
        fingers_pts = []
        for fingerLdm in multi_landmarks:
            mpDrawing.draw_landmarks(image, fingerLdm, mpHands.HAND_CONNECTIONS)
            for idx, lm in enumerate(fingerLdm.landmark):
                h, w, c = frame.shape()
                cx, cy = int(lm.x*w), int(lm.y*h)
                fingers_pts.append((cx, cy))
        for pts in fingers_pts:
            cv2.circle(frame, pts, 5, (128,128,128), 3, cv2.LINE_AA)
        upCount = 0
        for pts in fingers_coord:
            if fingers_pts[pts[0]][1] < fingers_pts[pts[1]][1]:
                upCount += 1
        if thumb_coord[pts[0]][0] > fingers_pts[pts[1]][0]:
            upCount += 1
        cv2.putText(frame, str(upCount), (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 12, (255, 255, 255), 5)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord(''):
        break
cap.release()
cv2.destroyAllWindows()