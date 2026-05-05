import cv2
import threading
import os
import time
import numpy as np
import platform as plt

class CustomVideoCapture():
    """客製化的影像擷取程式 - 用於實時從攝影機捕獲影像"""

    def __init__(self, dev=0):
        """初始化預設的攝影機裝置為 0（通常為內置攝影機）"""
        self.cap = cv2.VideoCapture(dev)
        self.ret = ''
        self.frame = []     
        self.win_title = 'Modified with set_title()'
        self.info = ''
        self.isStop = False
        self.t = threading.Thread(target=self.video, name='stream')

    def start_stream(self):
        """開啟執行緒開始串流攝影機影像"""
        self.t.start()
    
    def stop_stream(self):
        """關閉執行緒與攝影機"""
        self.isStop = True
        self.cap.release()
        cv2.destroyAllWindows()

    def get_current_frame(self):
        """取得最近一次的幀"""
        return self.ret, self.frame

    def set_title(self, txt):
        """設定顯示視窗的名稱"""
        self.win_title = txt

    def video(self):
        """執行緒主要運行的函式 - 持續捕獲影像並顯示"""
        global close_thread
        close_thread = 0

        while not self.isStop:
            # 從攝影機讀取一幀影像
            self.ret, self.frame = self.cap.read()
            
            # 水平翻轉影像（鏡像效果，讓使用者更直觀）
            self.frame = cv2.flip(self.frame, 1)

            # 若有資訊文字，則在影像上顯示
            if self.info != '':
                cv2.putText(self.frame, self.info, (10, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.namedWindow(self.win_title)
                cv2.imshow(self.win_title, self.frame)
                cv2.resizeWindow(self.win_title, 640, 480)

            # 檢查是否按下 ESC 鍵（27 是 ESC 的 ASCII 碼）
            if cv2.waitKey(1) == 27:
                break
            if close_thread == 1:
                break

        self.stop_stream()


def preprocess(frame, resize=(224, 224), norm=True):
    """前處理函式：將影像轉換為模型所需的格式
    
    處理流程：
    1. 中心裁切成正方形（保留寬高比）
    2. 調整大小到目標尺寸（224x224）
    3. 正規化像素值到 [-1, 1] 範圍
    4. 調整為 batch 格式 (1, 224, 224, 3)
    
    參數：
        frame: 輸入影像 (H, W, 3) - OpenCV 讀取的 BGR 影像
        resize: 目標尺寸，預設 (224, 224)
        norm: 是否進行正規化，預設 True
    
    回傳：
        input_format: 形狀為 (1, 224, 224, 3) 的 numpy 陣列，準備送入模型
    """
    height, width, _ = frame.shape
    
    # 步驟 1: 中心裁切成正方形
    # 選擇寬和高中較小的值作為裁切尺寸
    crop_size = min(width, height)
    # 計算裁切的起始位置，使其居中
    x_start = (width - crop_size) // 2
    y_start = (height - crop_size) // 2
    # 執行中心裁切
    cropped_image = frame[y_start:y_start + crop_size, x_start:x_start + crop_size]
    
    # 步驟 2: BGR → RGB 轉換
    # OpenCV 讀取的影像為 BGR，但 Teachable Machine 以 RGB 訓練，必須轉換
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

    # 步驟 3: 調整大小到目標尺寸
    # 使用雙線性插值調整大小，速度快且品質不錯
    resized_image = cv2.resize(cropped_image, resize, interpolation=cv2.INTER_LINEAR)

    # 步驟 4: 正規化像素值
    # 原始像素範圍：[0, 255]
    # 目標範圍：[-1, 1]（適合大多數深度學習模型）
    # 公式：(pixel / 127.5) - 1
    if norm:
        normalized_image = (resized_image.astype(np.float32) / 127.5) - 1.0
    else:
        normalized_image = resized_image.astype(np.float32)
    
    # 步驟 5: 調整為 batch 格式 (1, 224, 224, 3)
    # TensorFlow 期望的輸入格式為 (batch_size, height, width, channels)
    input_format = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    input_format[0] = normalized_image
    
    return input_format


def parse_output(preds, labels):
    """解析模型輸出結果
    
    參數：
        preds: 模型輸出的預測結果（概率分佈）
               形狀可能是 (1, num_classes) 或 (num_classes,)
        labels: 類別標籤列表
    
    回傳：
        (class_id, class_name, probability)：
            - class_id: 預測類別的索引
            - class_name: 預測類別的名稱
            - probability: 該類別的預測機率
    """
    # 處理不同形狀的輸出：若為 4D 張量則取第一個元素
    preds = preds[0] if len(preds.shape) == 4 else preds
    
    # 找出機率最高的類別索引
    trg_id = np.argmax(preds)
    # 根據索引查找類別名稱
    trg_name = labels[trg_id]
    # 取得該類別的預測機率
    trg_prob = preds[trg_id]
    
    return (trg_id, trg_name, trg_prob)


# 下方函式因為缺少 tensorrt 依賴，暫時保留但不使用
# def load_engine(engine_path):
#     """載入 TensorRT 引擎（需要先安裝 tensorrt 模組）"""
#     if trt_found:
#         TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
#         trt_runtime = trt.Runtime(TRT_LOGGER)
#         
#         with open(engine_path, 'rb') as f:
#             engine_data = f.read()
#         engine = trt_runtime.deserialize_cuda_engine(engine_data)
#         return engine
#     else:
#         print("Error: TensorRT module not found")
#         exit(1)