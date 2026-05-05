import cv2
import threading
import os
import time
import numpy as np
import tflite_runtime.interpreter as tflite
import tkinter as tk
from tkinter import messagebox
from tools import CustomVideoCapture, preprocess, parse_output

class ExerciseDetector:
    """運動動作識別類別"""
    
    # 配置常數
    MODEL_PATH = "model_colab.tflite"  # TensorFlow Lite 模型檔案路徑
    LABELS_PATH = "labels.txt"           # 標籤檔案路徑
    TRANSITION_THRESHOLD = 10  # 狀態轉換需要的連續穩定次數（幀數）
                               # 增加此值可降低誤判率，但會讓動作檢測變更遲鈍
                               # 例如：30fps 的視訊下，10 表示約需要 0.33 秒的穩定偵測
    MISS_THRESHOLD = 10         # 重置狀態前允許的連續錯誤次數
                               # 超過此次數會重置狀態機回到初始狀態
                               # 用於處理復雜背景或光線變化的干擾
    SLEEP_AFTER_COMPLETE = 2   # 完成所有動作後的延遲時間（秒），讓使用者看到結果
    SLEEP_BEFORE_EXIT = 1      # 程式結束前的延遲時間（秒）
    
    def __init__(self, model_path=None, labels_path=None):
        """初始化偵測器"""
        # 使用傳入的路徑，若未傳入則使用預設常數
        self.model_path = model_path or self.MODEL_PATH
        self.labels_path = labels_path or self.LABELS_PATH
        self.key_in = 0  # 使用者設定的目標次數
        self.interpreter = None
        self.labels = []
        self.vid = None
        
        # 狀態變數
        self.action_state = ["Relax", "Move", "Curl", "Move", "Relax"]
        self.current_state = 0
        self.curl_count = 0
        self.trans_count = 0
        self.miss_count = 0
        
        self._load_model()
        self._load_labels()
    
    def _load_model(self):
        """載入 TensorFlow Lite 模型"""
        try:
            self.interpreter = tflite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            print(f"✓ 模型載入成功: {self.model_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"模型檔案不存在: {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"模型載入失敗: {e}")
    
    def _load_labels(self):
        """載入標籤檔"""
        try:
            with open(self.labels_path, 'r') as f:
                self.labels = [line.strip().split()[1] for line in f.readlines()]
            print(f"✓ 標籤載入成功: {len(self.labels)} 個類別")
        except FileNotFoundError:
            raise FileNotFoundError(f"標籤檔案不存在: {self.labels_path}")
        except Exception as e:
            raise RuntimeError(f"標籤載入失敗: {e}")
    
    def get_input_output_details(self):
        """取得模型的輸入/輸出詳細資訊"""
        return self.interpreter.get_input_details(), self.interpreter.get_output_details()
    
    def show_gui(self):
        """顯示使用者輸入 GUI"""
        root = tk.Tk()
        root.title('Exercise Time')
        root.geometry('250x150')
        
        label = tk.Label(root, text='Number of times: ', font=("Arial", 14), padx=5, pady=5)
        label.grid(row=0, column=0)
        
        spinbox = tk.Spinbox(from_=0, to=100, width=5, font=("Arial", 14, "bold"))
        spinbox.grid(row=0, column=1)
        
        def on_confirm():
            try:
                self.key_in = int(spinbox.get()) if spinbox.get() else 0
                messagebox.showinfo('Confirm', f"You want to exercise for {self.key_in} times")
                root.destroy()
            except ValueError:
                messagebox.showerror('Error', "Please enter a valid number")
        
        button = tk.Button(root, text='  Enter  ', font=("Arial", 14, "bold"), 
                          command=on_confirm, background='#09c')
        button.grid(row=1, column=0, columnspan=2)
        
        root.mainloop()
    
    def process_frame(self, frame):
        """處理單一影格：前處理 -> 推論 -> 輸出解析
        
        流程說明：
        1. 前處理：縮放到 224x224 並進行正規化，符合模型輸入要求
        2. 設定輸入 tensor 並執行推論
        3. 取得輸出 tensor（各類別的機率分佈）
        4. 解析輸出得到最高機率的類別、ID 和信心度
        """
        # 將輸入影像調整為模型要求的尺寸並進行正規化
        data = preprocess(frame, resize=(224, 224), norm=True)
        
        input_details, output_details = self.get_input_output_details()
        # 將前處理後的影像設定為輸入 tensor
        self.interpreter.set_tensor(input_details[0]['index'], data)
        # 執行推論
        self.interpreter.invoke()
        
        # 取得推論結果（輸出 tensor）：包含各類別的機率分佈
        prediction = self.interpreter.get_tensor(output_details[0]['index'])[0]
        # 解析輸出為 (類別ID, 類別名稱, 信心度)
        trg_id, trg_class, trg_prob = parse_output(prediction, self.labels)
        
        return trg_class, trg_prob
    
    def update_state(self, detected_class):
        """更新狀態機：判斷是否轉換狀態"""
        if detected_class == self.action_state[self.current_state]:
            # 符合當前期望動作，維持狀態
            self.miss_count = 0
            self.trans_count = 0  # 重置轉換計數，確保需要連續穩定偵測才能轉換
            
        elif (self.current_state < len(self.action_state) - 1 and 
              detected_class == self.action_state[self.current_state + 1]):
            # 偵測到下一個期望動作
            self.trans_count += 1
            # 需要連續穩定多次才轉換狀態，避免單次誤判導致狀態跳轉
            # 例如：TRANSITION_THRESHOLD=10 表示需要連續 10 幀都正確偵測到下一個動作
            if self.trans_count > self.TRANSITION_THRESHOLD:
                self.current_state += 1
                self.trans_count = 0
                
                # 檢查是否已到達動作序列的最後一步，若是則完成一次完整動作計數
                # 例如：actionState = ["Relax", "Move", "Curl", "Move", "Relax"]
                # 當 current_state == 4（最後一個）時，表示已完成從開始到結束的整個序列
                if self.current_state == len(self.action_state) - 1:
                    self.current_state = 0  # 重置狀態機以計算下一次
                    self.curl_count += 1
                    print(f"✓ 完成第 {self.curl_count} 次動作")
        else:
            # 不符合期望的動作，增加錯誤計數
            self.miss_count += 1
            # 若連續錯誤次數超過閾值（MISS_THRESHOLD=5），則重置狀態機
            # 這樣做可以防止在複雜背景或光線不佳時卡在錯誤的狀態
            if self.miss_count > self.MISS_THRESHOLD:
                self.current_state = 0  # 重置回初始狀態
                self.trans_count = 0    # 清空轉換計數
                self.miss_count = 0     # 清空錯誤計數
                print(f"⚠ 狀態重置（連續 {self.MISS_THRESHOLD} 幀偵測錯誤）")
    
    def run(self):
        """主執行迴圈
        
        工作流程：
        1. 顯示 GUI 讓使用者輸入目標動作次數
        2. 初始化攝影機串流
        3. 逐幀讀取影像、推論、更新狀態機
        4. 當達到目標次數時結束
        5. 清理資源
        """
        self.show_gui()
        
        # 驗證使用者輸入的有效性
        if self.key_in <= 0:
            print("✗ 無效的次數設定，結束程式")
            return
        
        try:
            self.vid = CustomVideoCapture()
            self.vid.set_title('Health Promotion')
            self.vid.start_stream()
            
            print(f"✓ 開始偵測，目標: {self.key_in} 次")
            
            # 主偵測迴圈：持續讀取影格直到達到目標次數或使用者停止
            while not self.vid.isStop:
                ret, frame = self.vid.get_current_frame()
                if not ret:
                    continue
                
                # 推論當前影格得到偵測到的動作類別和信心度
                detected_class, prob = self.process_frame(frame)
                # 根據偵測結果更新狀態機
                self.update_state(detected_class)
                
                # 在視窗中顯示即時資訊：當前動作、信心度、計數進度
                self.vid.info = f'{detected_class} (Conf: {prob:.2f}), Count: {self.curl_count}/{self.key_in}'
                
                # 檢查是否已完成指定次數的動作
                if self.curl_count == self.key_in:
                    print(f"✓ 完成 {self.curl_count} 次動作")
                    time.sleep(self.SLEEP_AFTER_COMPLETE)
                    break
            
            # 顯示最終結果資訊
            self.vid.info = f'Finish {self.curl_count} times curls. Press Esc to exit.'
            time.sleep(self.SLEEP_BEFORE_EXIT)
            
        except Exception as e:
            print(f"✗ 執行錯誤: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理資源"""
        if self.vid:
            self.vid.stop_stream()
        print("✓ 程式結束")

if __name__ == "__main__":
    # 可選：自訂模型和標籤檔案路徑
    # detector = ExerciseDetector(model_path="custom_model.tflite", labels_path="custom_labels.txt")
    
    # 使用預設路徑
    detector = ExerciseDetector()
    detector.run()