from random import randint

symbols = ['*','#','@','$']
# 2026-05-05 修正：改用 with 語句，確保檔案在任何情況下都會自動關閉，避免資源洩漏
with open("symbols.txt", "w") as fd:
    for i in range(20):
        for j in range(randint(1,30)):
            fd.write(symbols[randint(0,len(symbols)-1)])
        fd.write("\n")
