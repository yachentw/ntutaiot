import asyncio
from bleak import BleakScanner

async def main():
    print("Scanning 10 seconds..")
    
    # === 對應原本的 scanner.scan(10.0) ===
    # return_adv=True 會同時回傳「裝置資訊」與「廣播封包(含RSSI/學號)」
    # 這會是一個 Dictionary: { MAC地址: (device, adv_data) }
    scanned_results = await BleakScanner.discover(timeout=10.0, return_adv=True)
    
    print(f"Scan finished, found {len(scanned_results)} devices.\n")

    # === 對應原本的 for dev in devices: ===
    for _, (device, adv_data) in scanned_results.items():

        # 檢查是否有廠商資料 (Manufacturer Data)
        if adv_data.manufacturer_data:
            # 遍歷這台裝置的所有廠商 ID
            for m_id, m_data in adv_data.manufacturer_data.items():

                # === 解析各廠商格式，產生 label 與 extra 欄位 ===
                label = None
                extra = {}

                # Android 自訂廣播 (0xFFFF)
                if m_id == 0xFFFF:
                    label = "Android Beacon"
                    extra["Data"] = f"0x{m_data.hex()}"

                # Apple iBeacon (nRF Connect on iPhone, 0x004C)
                # iBeacon 格式: byte0=0x02 (type), byte1=0x15 (length=21),
                #               bytes2-17=UUID, bytes18-19=Major, bytes20-21=Minor, byte22=TxPower
                elif m_id == 0x004C and len(m_data) >= 23 and m_data[0] == 0x02 and m_data[1] == 0x15:
                    uuid_bytes = m_data[2:18]
                    label = "Apple iBeacon (nRF Connect)"
                    extra["UUID"]    = (
                        f"{uuid_bytes[0:4].hex()}-{uuid_bytes[4:6].hex()}-"
                        f"{uuid_bytes[6:8].hex()}-{uuid_bytes[8:10].hex()}-"
                        f"{uuid_bytes[10:16].hex()}"
                    )
                    extra["Major"]   = int.from_bytes(m_data[18:20], "big")
                    extra["Minor"]   = int.from_bytes(m_data[20:22], "big")
                    extra["TxPower"] = f"{int.from_bytes(m_data[22:23], 'big', signed=True)} dBm"

                # === 共用輸出區塊 ===
                if label:
                    name = device.name if device.name else "Unknown"
                    print("-" * 40)
                    print(f"[★] Found {label}！")
                    print(f"Device : {name}")
                    print(f"Addr   : {device.address}")
                    print(f"RSSI   : {adv_data.rssi} dBm")
                    for k, v in extra.items():
                        print(f"{k:<7}: {v}")
                    print("-" * 40)

if __name__ == "__main__":
    asyncio.run(main())