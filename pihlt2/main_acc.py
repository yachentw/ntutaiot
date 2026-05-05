import os
import threading
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from queue import Queue
from tflite_runtime.interpreter import Interpreter
import mpu6050_raw
# 2026-05-05 加速：改用 fig.canvas.buffer_rgba() 直接取像素，移除 io / PIL 依賴

# 2026-05-05 修正：使用腳本所在目錄作為基準，避免相對路徑受執行目錄影響
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS_PATH = os.path.join(SCRIPT_DIR, "accel_labels.txt")
MODEL_PATH = os.path.join(SCRIPT_DIR, "accel_model.tflite")

# 可調整參數
sampling_interval = 0.01  # 感測器取樣間隔（秒），決定資料解析度
window_size = 0.5          # 每次分析的時間窗口大小（秒）
slide_interval = 0.1       # 滑動視窗的移動間隔（秒）
samples_num = int(window_size // sampling_interval)  # 每個 window 包含的樣本數

TRANSITION_THRESHOLD = 1  # 狀態轉換需要的連續穩定次數（window 數）
                          # 每個 window = 0.5 秒，故 1 表示約需 1 秒穩定偵測
MISS_THRESHOLD = 2        # 重置狀態前允許的連續錯誤次數
                          # 超過此次數會重置狀態機回到初始狀態

# 狀態機變數
currentState = 0          # 目前所在的動作狀態索引
Curl_Count = 0            # 已完成的動作次數
transCount = 0            # 連續偵測到下一個狀態的次數（防抖用）
missCount = 0             # 連續偵測錯誤的次數（錯誤恢復用）
with open(LABELS_PATH, 'r') as f:
    labels = [line.strip().split()[-1] for line in f]   # 從檔案載入類別名稱
# actionState = ["Relax", "Move", "Curl", "Move", "Relax"]  # 完整動作序列（保留備用）
actionState = ["Relax", "Curl", "Relax"]             # 簡化動作序列：放鬆 → 動作 → 放鬆


def initialize_plot():
    """初始化 matplotlib 圖表

    figsize=(2.24, 2.24) 搭配預設 DPI=100，產生剛好 224×224 像素的圖表，
    符合 Teachable Machine 模型的輸入尺寸要求。
    """
    fig, ax = plt.subplots(figsize=(2.24, 2.24))
    accel_lines = [
        ax.plot([], [], label='Accel X')[0],
        ax.plot([], [], label='Accel Y')[0],
        ax.plot([], [], label='Accel Z')[0]
    ]
    # 隱藏 X 軸刻度，Y 軸固定在加速度正規化範圍 [-1.25, 1.25]
    ax.xaxis.set_visible(False)
    ax.set_ylim(-1.25, 1.25)

    plt.tight_layout()
    return fig, ax, accel_lines


def update_plot(fig, accel_lines, data_points):
    """更新圖表資料

    2026-05-05 修正：移除無效的 blit 邏輯。Agg backend 不支援 blit/flush_events，
    實際渲染由主迴圈的 fig.canvas.draw() 負責，此函式只負責更新折線資料。
    """
    timestamps = [point['time'] for point in data_points]
    accel_data = [
        [point['accel']['x'] for point in data_points],
        [point['accel']['y'] for point in data_points],
        [point['accel']['z'] for point in data_points]
    ]

    for line, data in zip(accel_lines, accel_data):
        line.set_data(timestamps, data)

    fig.axes[0].set_xlim(timestamps[0], timestamps[-1])


def sensor_reader(queue, stop_event):
    """感測器讀取執行緒：持續從 MPU6050 取得加速度資料並放入佇列

    2026-05-05 加速：移除推論期間暫停機制，感測器持續運行，
    推論結束後清空佇列舊資料，確保下個 window 使用最新資料。
    """
    start_time = time.time()
    # 2026-05-05 加速：改為 while not stop_event，stop_event 僅用於程式結束
    while not stop_event.is_set():
        try:
            accel, _ = mpu6050_raw.getAccelGyro()  # 只取加速度，忽略陀螺儀
            timestamp = time.time()
            elapsed_time = timestamp - start_time
            queue.put({'time': elapsed_time, 'accel': accel})
            time.sleep(sampling_interval)
        except Exception as e:
            print(f"Error in sensor_reader: {e}")
            break


def preprocess(frame, out, norm=True):
    """前處理函式：將圖表影像轉換為模型所需格式

    輸入來源為 fig.canvas.buffer_rgba() 取得的 RGBA numpy 陣列，
    已是 RGB 順序，不需 BGR→RGB 轉換，也不需 resize（figsize 已控制尺寸）。

    2026-05-05 加速：接收外部預先分配的 out buffer，避免每次推論重新配置記憶體。
    正規化公式：(pixel / 127.5) - 1，將 [0, 255] 映射到 [-1, 1]
    """
    # 取前三個通道（R, G, B），去除 RGBA 的 alpha 通道
    frame_rgb = frame[:, :, :3]

    # 正規化並原地寫入預先分配的 buffer
    if norm:
        out[0] = (frame_rgb.astype(np.float32) / 127.5) - 1.0
    else:
        out[0] = frame_rgb.astype(np.float32)
    return out


if __name__ == "__main__":
    mpu6050_raw.init()
    data_queue = Queue()
    stop_event = threading.Event()

    # 載入 TensorFlow Lite 模型並配置輸入/輸出 tensor
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    try:
        stop_event.clear()
        # 以 daemon 執行緒運行感測器讀取，主程式結束時自動終止
        sensor_thread = threading.Thread(target=sensor_reader, args=(data_queue, stop_event), daemon=True)
        sensor_thread.start()

        fig, ax, accel_lines = initialize_plot()
        # 2026-05-05 修正：移除 plt.ion()，Agg backend 不支援互動模式

        # 2026-05-05 加速：預先分配 input buffer，避免每次推論重新配置記憶體
        input_buffer = np.empty((1, 224, 224, 3), dtype=np.float32)

        recorded_data = []
        while True:
            # 累積足夠的樣本數（一個完整 window）
            while len(recorded_data) < samples_num:
                recorded_data.append(data_queue.get())

            if len(recorded_data) >= samples_num:
                # 將感測器資料繪製成圖表
                update_plot(fig, accel_lines, recorded_data[:samples_num])

                # 2026-05-05 加速：直接從 canvas buffer 取像素，跳過 PNG 編解碼
                # 使用 np.asarray 相容 matplotlib 各版本（memoryview / ndarray 皆可）
                fig.canvas.draw()
                w, h = fig.canvas.get_width_height()
                frame = np.asarray(fig.canvas.buffer_rgba()).reshape(h, w, 4)

                # 前處理後送入模型推論（使用預先分配的 buffer）
                input_data = preprocess(frame, out=input_buffer)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                prediction = interpreter.get_tensor(output_details[0]['index'])[0]

                # 取得機率最高的類別作為本次 window 的辨識結果
                trg_class = labels[np.argmax(prediction)]
                print(f"Prediction: {trg_class} ({prediction})")
                print(trg_class, actionState[currentState])

                # 狀態機：根據辨識結果決定是否轉換狀態
                if trg_class == actionState[currentState]:
                    # 符合當前狀態，重置計數器
                    transCount = 0
                    missCount = 0
                elif currentState < len(actionState) - 1 and trg_class == actionState[currentState + 1]:
                    # 偵測到下一個期望動作，累積轉換計數
                    transCount += 1
                    # 連續穩定偵測超過閾值才轉換，避免單次誤判
                    if transCount > TRANSITION_THRESHOLD:
                        currentState += 1
                        transCount = 0
                        missCount = 0
                        # 到達動作序列末端，計算完成一次並重置
                        if currentState == len(actionState) - 1:
                            currentState = 0
                            Curl_Count += 1
                else:
                    # 非預期動作，累積錯誤計數
                    missCount += 1
                    # 連續錯誤超過閾值，重置狀態機回到初始狀態
                    if missCount > MISS_THRESHOLD:
                        missCount = 0
                        transCount = 0
                        currentState = 0

                print("currentState:", currentState)
                print("curl count: ", Curl_Count)

                # 2026-05-05 加速：感測器持續運行，推論結束後清空佇列舊資料
                recorded_data = []
                with data_queue.mutex:
                    data_queue.queue.clear()

    except KeyboardInterrupt:
        print("\nExiting program.")
        stop_event.set()
        # 2026-05-05 修正：移除 plt.ioff() 和 plt.show()，Agg backend 不會顯示視窗
