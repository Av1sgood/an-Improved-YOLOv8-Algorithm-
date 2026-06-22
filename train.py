import ultralytics.nn.tasks as tasks
from ultralytics import YOLO

# 匯入我們的自定義模組
from custom_modules import C2f_RFAConv, GSConv, VoV_GSCSP, ASFF_Detect

# ==========================================
# 模組身份劫持 (Identity Hijacking)
# ==========================================
tasks.C3x = C2f_RFAConv        
tasks.C3Ghost = VoV_GSCSP      
tasks.GhostConv = GSConv       
# 核心修正：將官方的 Detect 替換為我們加入 ASFF 的版本
tasks.Detect = ASFF_Detect  

print("[-] Four-Detect-ASFF 劫持註冊完成，準備啟動訓練。")

if __name__ == '__main__':
    model = YOLO("yolov8_water.yaml")
    
    model.train(data=r'/media/ab12good/2cafb626-c66c-4dcf-9b40-9ffb300ce999/yolov8_Reflect/Floating Garbage Detection.yolov8/data.yaml',
        epochs=200, 
        imgsz=640, 
        batch=16, 
        device=0
    )
