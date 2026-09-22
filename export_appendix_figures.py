"""Xuất 4 hình phụ lục minh hoạ landmark MediaPipe (B1-B4) từ webcam.

Chạy: python export_appendix_figures.py  ->  chọn 1-4, nhấn S để lưu, Q để thoát.
"""
import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

# mode -> (ten file, cac chi so landmark, mau BGR). indices=None nghia la ve toan bo mesh.
FIGURES = {
    "1": ("Figure_B1_FaceMesh.png",  None,                                        None),
    "2": ("Figure_B2_EAR.png",       {33, 160, 158, 133, 153, 144,
                                      263, 387, 385, 362, 380, 373},             (0, 0, 255)),
    "3": ("Figure_B3_MAR.png",       {78, 81, 13, 308, 14, 312},                  (0, 255, 0)),
    "4": ("Figure_B4_HeadPose.png",  {1, 33, 61, 152, 263, 291},                  (255, 0, 0)),
}


def draw_points(image, face, indices, color):
    """Vẽ chấm + số thứ tự cho các landmark được chọn."""
    h, w, _ = image.shape
    for idx in indices:
        lm = face.landmark[idx]
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (x, y), 4, color, -1)
        cv2.putText(image, str(idx), (x + 4, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def main():
    print("=" * 50)
    print("Appendix Figure Generator")
    print("=" * 50)
    for key, (filename, _, _) in FIGURES.items():
        print(f"{key} : {filename}")
    print("S để lưu | Q để thoát")

    mode = input("Chọn hình (1-4): ").strip()
    if mode not in FIGURES:
        return print(f"Lựa chọn không hợp lệ: {mode!r}")

    filename, indices, color = FIGURES[mode]
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            output = frame.copy()
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.multi_face_landmarks:
                face = results.multi_face_landmarks[0]
                if indices is None:
                    mp_draw.draw_landmarks(
                        image=output, landmark_list=face,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_style.get_default_face_mesh_tesselation_style())
                else:
                    draw_points(output, face, indices, color)

            cv2.imshow("Appendix Figure Generator", output)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                cv2.imwrite(filename, output)
                print(f"Đã lưu {filename}")
            elif key == ord("q"):
                break
    finally:
        cap.release()
        face_mesh.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
