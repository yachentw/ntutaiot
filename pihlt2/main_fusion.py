"""
main_fusion.py
2026-05-05 新增：攝影機（lec08/main.py）與加速度計（main_acc.py）的
Decision-level Fusion。兩個模型各自在獨立執行緒推論，
只有兩者都同意同一動作時才更新狀態機，降低誤判率。
"""

import os
import sys
import cv2
import threading
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from queue import Queue, Empty
from tflite_runtime.interpreter import Interpreter
import mpu6050_raw
# 2026-05-05 加速：改用 fig.canvas.buffer_rgba() 直接取像素，移除 io / PIL 依賴

# 2026-05-05 修正：使用腳本所在目錄作為基準，避免相對路徑受執行目錄影響
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 模型路徑設定 ──────────────────────────────────────────────────────────────
CAMERA_MODEL_PATH  = os.path.join(SCRIPT_DIR, "camera_model.tflite")
CAMERA_LABELS_PATH = os.path.join(SCRIPT_DIR, "camera_labels.txt")
ACCEL_MODEL_PATH   = os.path.join(SCRIPT_DIR, "accel_model.tflite")
ACCEL_LABELS_PATH  = os.path.join(SCRIPT_DIR, "accel_labels.txt")

# ── 動作序列與狀態機參數 ──────────────────────────────────────────────────────
ACTION_STATE         = ["Relax", "Curl", "Relax"]
TRANSITION_THRESHOLD = 1   # 連續幾個 window 都同意才轉換狀態
MISS_THRESHOLD       = 3   # 連續幾個 window 都誤判才重置狀態機

# ── 加速度計參數 ──────────────────────────────────────────────────────────────
SAMPLING_INTERVAL = 0.01                               # 感測器取樣間隔（秒）
WINDOW_SIZE       = 0.5                                # 分析時間窗口（秒）
SAMPLES_NUM       = int(WINDOW_SIZE // SAMPLING_INTERVAL)
ACCEL_LABELS      = []   # 從 accel_labels.txt 載入

# ── 融合信心度閾值 ────────────────────────────────────────────────────────────
CONFIDENCE_OVERRIDE = 0.9   # 單一感測器信心度超過此值時可獨立決策


# ── 共享預測狀態（執行緒安全） ────────────────────────────────────────────────
class SharedPrediction:
    """存放兩個模態的最新預測，提供執行緒安全的讀寫介面"""

    def __init__(self):
        self._lock        = threading.Lock()
        self.camera_class = None
        self.camera_conf  = 0.0
        self.accel_class  = None
        self.accel_conf   = 0.0

    def update_camera(self, cls, conf):
        with self._lock:
            self.camera_class = cls
            self.camera_conf  = conf

    def update_accel(self, cls, conf):
        with self._lock:
            self.accel_class = cls
            self.accel_conf  = conf

    def get(self):
        with self._lock:
            return (self.camera_class, self.camera_conf,
                    self.accel_class,  self.accel_conf)


# ── 影像前處理 ────────────────────────────────────────────────────────────────
def preprocess_camera(frame, out, resize=(224, 224)):
    """BGR 影像 → 模型輸入 (1,224,224,3)：中心裁切 → BGR→RGB → 正規化

    2026-05-05 加速：接收外部預先分配的 out buffer，避免每幀重新配置記憶體。
    """
    h, w, _ = frame.shape
    crop = min(h, w)
    y0, x0 = (h - crop) // 2, (w - crop) // 2
    cropped = frame[y0:y0+crop, x0:x0+crop]
    rgb     = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, resize, interpolation=cv2.INTER_LINEAR)
    out[0]  = (resized.astype(np.float32) / 127.5) - 1.0
    return out


def preprocess_accel(fig, out):
    """matplotlib 圖表 → 模型輸入 (1,224,224,3)

    2026-05-05 加速：直接從 canvas buffer 取像素，跳過 PNG 編解碼；
    使用 np.asarray 相容 matplotlib 各版本（memoryview / ndarray 皆可）；
    使用外部預先分配的 out buffer，避免每個 window 重新配置記憶體。
    """
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    frame = np.asarray(fig.canvas.buffer_rgba()).reshape(h, w, 4)
    out[0] = (frame[:, :, :3].astype(np.float32) / 127.5) - 1.0
    return out


# ── 融合決策函式 ──────────────────────────────────────────────────────────────
def fuse(cam_cls, cam_conf, acc_cls, acc_conf):
    """
    Decision-level Fusion 規則（依優先順序）：
    1. 兩者預測一致          → 採用，信心度取平均
    2. 攝影機信心度 >= 閾值   → 採用攝影機（感測器良好時的快速決策）
    3. 加速度計信心度 >= 閾值 → 採用加速度計
    4. 其他                  → 回傳 None（不確定，不更新狀態機）
    """
    if cam_cls == acc_cls:
        return cam_cls, (cam_conf + acc_conf) / 2
    if cam_conf >= CONFIDENCE_OVERRIDE:
        return cam_cls, cam_conf
    if acc_conf >= CONFIDENCE_OVERRIDE:
        return acc_cls, acc_conf
    return None, 0.0


# ── 攝影機推論執行緒 ──────────────────────────────────────────────────────────
def camera_worker(shared: SharedPrediction, labels: list,
                  interpreter: Interpreter, stop_event: threading.Event):
    """持續從攝影機讀取影格，執行推論並更新 shared.camera_*"""
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 2026-05-05 加速：預先分配 input buffer，避免每幀重新配置記憶體
    input_buffer = np.empty((1, 224, 224, 3), dtype=np.float32)

    cap = cv2.VideoCapture(0)
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)

            interpreter.set_tensor(input_details[0]['index'],
                                   preprocess_camera(frame, out=input_buffer))
            interpreter.invoke()
            pred = interpreter.get_tensor(output_details[0]['index'])[0]

            idx  = int(np.argmax(pred))
            cls  = labels[idx]
            conf = float(pred[idx])
            shared.update_camera(cls, conf)

            # 顯示攝影機畫面與目前預測
            cam_cls, cam_conf, acc_cls, acc_conf = shared.get()
            fused_cls, fused_conf = fuse(cam_cls, cam_conf, acc_cls, acc_conf)
            info = (f"CAM:{cam_cls}({cam_conf:.2f})  "
                    f"ACC:{acc_cls}({acc_conf:.2f})  "
                    f"FUSED:{fused_cls}({fused_conf:.2f})")
            cv2.putText(frame, info, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow('Fusion', frame)
            if cv2.waitKey(1) == 27:   # ESC 結束
                stop_event.set()
    finally:
        cap.release()
        cv2.destroyAllWindows()


# ── 加速度計感測器讀取執行緒 ──────────────────────────────────────────────────
def sensor_reader(data_queue: Queue, stop_event: threading.Event):
    """持續讀取 MPU6050 資料並放入佇列"""
    start = time.time()
    while not stop_event.is_set():
        try:
            accel, _ = mpu6050_raw.getAccelGyro()
            data_queue.put({'time': time.time() - start, 'accel': accel})
            time.sleep(SAMPLING_INTERVAL)
        except Exception as e:
            print(f"[感測器] 讀取錯誤: {e}")
            break


# ── 加速度計推論執行緒 ────────────────────────────────────────────────────────
def accel_worker(shared: SharedPrediction, interpreter: Interpreter,
                 stop_event: threading.Event):
    """每個 window（0.5 秒）執行一次推論並更新 shared.accel_*"""
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    data_queue = Queue()
    sensor_thread = threading.Thread(
        target=sensor_reader, args=(data_queue, stop_event), daemon=True)
    sensor_thread.start()

    # figsize=(2.24, 2.24) × DPI=100 → 224×224 像素，符合模型輸入
    fig, ax = plt.subplots(figsize=(2.24, 2.24))
    lines = [ax.plot([], [], label=lbl)[0] for lbl in ['X', 'Y', 'Z']]
    ax.xaxis.set_visible(False)
    ax.set_ylim(-1.25, 1.25)
    plt.tight_layout()

    # 2026-05-05 加速：預先分配 input buffer，避免每個 window 重新配置記憶體
    input_buffer = np.empty((1, 224, 224, 3), dtype=np.float32)

    buffer = []
    while not stop_event.is_set():
        try:
            buffer.append(data_queue.get(timeout=1.0))
        except Empty:
            continue

        if len(buffer) < SAMPLES_NUM:
            continue

        window = buffer[:SAMPLES_NUM]
        buffer = []

        # 更新圖表
        ts = [p['time'] for p in window]
        for i, key in enumerate(['x', 'y', 'z']):
            lines[i].set_data(ts, [p['accel'][key] for p in window])
        ax.set_xlim(ts[0], ts[-1])

        # 推論
        interpreter.set_tensor(input_details[0]['index'],
                                preprocess_accel(fig, out=input_buffer))
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])[0]

        idx  = int(np.argmax(pred))
        cls  = ACCEL_LABELS[idx]
        conf = float(pred[idx])
        shared.update_accel(cls, conf)

    plt.close(fig)


# ── 融合狀態機（主執行緒） ────────────────────────────────────────────────────
def run_state_machine(shared: SharedPrediction,
                      target_count: int,
                      stop_event: threading.Event):
    """每個 window 週期執行一次融合判斷，驅動狀態機計數"""
    state       = 0
    curl_count  = 0
    trans_count = 0
    miss_count  = 0

    print(f"開始偵測，目標：{target_count} 次")
    while not stop_event.is_set():
        time.sleep(WINDOW_SIZE)   # 與加速度計 window 同步

        cam_cls, cam_conf, acc_cls, acc_conf = shared.get()
        if cam_cls is None or acc_cls is None:
            continue   # 兩個模型尚未完成第一次預測

        fused_cls, fused_conf = fuse(cam_cls, cam_conf, acc_cls, acc_conf)
        expected      = ACTION_STATE[state]
        next_expected = (ACTION_STATE[state + 1]
                         if state < len(ACTION_STATE) - 1 else None)

        print(f"[融合] CAM={cam_cls}({cam_conf:.2f}) "
              f"ACC={acc_cls}({acc_conf:.2f}) "
              f"→ {fused_cls}({fused_conf:.2f})  "
              f"期望:{expected}  計數:{curl_count}/{target_count}")

        if fused_cls == expected:
            # 符合當前狀態，重置計數器
            trans_count = 0
            miss_count  = 0
        elif fused_cls == next_expected:
            # 偵測到下一個期望動作，累積轉換計數
            trans_count += 1
            if trans_count > TRANSITION_THRESHOLD:
                state      += 1
                trans_count = 0
                miss_count  = 0
                # 到達動作序列末端，完成一次計數
                if state == len(ACTION_STATE) - 1:
                    state       = 0
                    curl_count += 1
                    print(f"✓ 完成第 {curl_count} 次動作")
                    if curl_count >= target_count:
                        print(f"✓ 完成目標 {target_count} 次！")
                        stop_event.set()
                        break
        else:
            # 非預期動作，累積錯誤計數
            miss_count += 1
            if miss_count > MISS_THRESHOLD:
                state       = 0
                trans_count = 0
                miss_count  = 0
                print("⚠ 狀態重置（連續誤判）")


# ── 主程式 ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 載入攝影機標籤與模型
    try:
        with open(CAMERA_LABELS_PATH, 'r') as f:
            cam_labels = [line.strip().split()[-1] for line in f]
        cam_interpreter = Interpreter(model_path=CAMERA_MODEL_PATH)
        cam_interpreter.allocate_tensors()
        print(f"✓ 攝影機模型載入：{CAMERA_MODEL_PATH}")
    except FileNotFoundError as e:
        print(f"✗ {e}")
        sys.exit(1)

    # 載入加速度計模型與標籤
    try:
        with open(ACCEL_LABELS_PATH, 'r') as f:
            ACCEL_LABELS.extend(line.strip().split()[-1] for line in f)
        acc_interpreter = Interpreter(model_path=ACCEL_MODEL_PATH)
        acc_interpreter.allocate_tensors()
        print(f"✓ 加速度計模型載入：{ACCEL_MODEL_PATH}")
    except FileNotFoundError as e:
        print(f"✗ {e}")
        sys.exit(1)

    # 初始化 MPU6050
    mpu6050_raw.init()
    print("✓ MPU6050 初始化完成")

    try:
        target = int(input("目標次數："))
    except ValueError:
        print("✗ 請輸入有效整數")
        sys.exit(1)
    if target <= 0:
        print("✗ 次數必須大於 0")
        sys.exit(1)

    shared     = SharedPrediction()
    stop_event = threading.Event()

    cam_thread = threading.Thread(
        target=camera_worker,
        args=(shared, cam_labels, cam_interpreter, stop_event),
        daemon=True, name='camera')
    acc_thread = threading.Thread(
        target=accel_worker,
        args=(shared, acc_interpreter, stop_event),
        daemon=True, name='accel')

    cam_thread.start()
    acc_thread.start()

    try:
        run_state_machine(shared, target, stop_event)
    except KeyboardInterrupt:
        print("\n使用者中斷")
    finally:
        stop_event.set()
        cam_thread.join(timeout=3)
        acc_thread.join(timeout=3)
        print("✓ 程式結束")
