import cv2

cam = cv2.VideoCapture(0)

# 2026-05-05 修正：加入 ret 檢查與 try-finally，避免讀取失敗時對 None 操作崩潰，
# 並確保 cam.release() 在任何情況下都會執行
try:
    ret, image = cam.read()
    if not ret:
        print("Error: Failed to capture image.")
    else:
        cv2.imshow('preview', image)
        cv2.waitKey(0)
        cv2.imwrite('/home/pi/cvimage.jpg', image)
finally:
    cam.release()
    cv2.destroyAllWindows()
