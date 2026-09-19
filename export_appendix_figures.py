import cv2
import mediapipe as mp

# ==========================
# Landmark definitions
# ==========================

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]
MOUTH = [78, 81, 13, 308, 14, 312]
HEAD_POSE = [1, 33, 61, 152, 263, 291]

# ==========================
# MediaPipe
# ==========================

mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

print("=" * 50)
print("Appendix Figure Generator")
print("=" * 50)
print("1 : Face Mesh")
print("2 : EAR Landmarks")
print("3 : MAR Landmarks")
print("4 : Head Pose Landmarks")
print("Press S to save")
print("Press Q to quit")

mode = input("Choose figure (1-4): ")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    output = frame.copy()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        face = results.multi_face_landmarks[0]

        h, w, _ = frame.shape

        # -------------------------
        # Figure B1
        # -------------------------

        if mode == "1":

            mp_draw.draw_landmarks(
                image=output,
                landmark_list=face,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_style.get_default_face_mesh_tesselation_style()
            )

        # -------------------------
        # Figure B2
        # -------------------------

        elif mode == "2":

            for idx, lm in enumerate(face.landmark):

                if idx in LEFT_EYE or idx in RIGHT_EYE:

                    x = int(lm.x * w)
                    y = int(lm.y * h)

                    cv2.circle(output, (x, y), 4, (0, 0, 255), -1)

                    cv2.putText(
                        output,
                        str(idx),
                        (x + 4, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1
                    )

        # -------------------------
        # Figure B3
        # -------------------------

        elif mode == "3":

            for idx, lm in enumerate(face.landmark):

                if idx in MOUTH:

                    x = int(lm.x * w)
                    y = int(lm.y * h)

                    cv2.circle(output, (x, y), 4, (0, 255, 0), -1)

                    cv2.putText(
                        output,
                        str(idx),
                        (x + 4, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1
                    )

        # -------------------------
        # Figure B4
        # -------------------------

        elif mode == "4":

            for idx, lm in enumerate(face.landmark):

                if idx in HEAD_POSE:

                    x = int(lm.x * w)
                    y = int(lm.y * h)

                    cv2.circle(output, (x, y), 5, (255, 0, 0), -1)

                    cv2.putText(
                        output,
                        str(idx),
                        (x + 4, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1
                    )

    cv2.imshow("Appendix Figure Generator", output)

    key = cv2.waitKey(1)

    if key == ord("s"):

        if mode == "1":
            filename = "Figure_B1_FaceMesh.png"

        elif mode == "2":
            filename = "Figure_B2_EAR.png"

        elif mode == "3":
            filename = "Figure_B3_MAR.png"

        elif mode == "4":
            filename = "Figure_B4_HeadPose.png"

        cv2.imwrite(filename, output)

        print(f"Saved {filename}")

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
