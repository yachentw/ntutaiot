import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD)
BTN_PIN = 11
GPIO.setup(BTN_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
previousStatus = None

try:
    while True:
        input = GPIO.input(BTN_PIN)
        if input == GPIO.LOW and previousStatus == GPIO.HIGH:
            print("Button pressed @", time.ctime())
        previousStatus = input
        time.sleep(0.01)  # 2026-05-05 修正：輪詢加入延遲，避免 CPU 100% 空轉
except KeyboardInterrupt:
    print("Exception: KeyboardInterrupt")

finally:
    GPIO.cleanup() 
