import os
import random
import shutil
from pathlib import Path

# =====================================================================
# 1. 基本路徑與參數設定
# =====================================================================
DATASET_ROOT = "/mnt/data/AI_project/paper/dataset"      # 您上傳的原始資料集總目錄
OUTPUT_ROOT = "/mnt/data/AI_project/paper/dataset_split"   # 重新劃分後的全新資料集輸出路徑

TRAIN_RATIO = 0.8  # 訓練集比例
VAL_RATIO = 0.2    # 驗證集比例

# 支援的影像副檔名格式 (注解：副檔名格式)
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.jpg', '.jpeg', '.png')

# =====================================================================
# 2. 使用 os.walk 遞迴搜尋所有子資料夾內的影像
# =====================================================================
all_images = []

print("[-] 開始深入搜尋所有子資料夾...")

# os.walk 會遍歷根目錄下的所有子目錄
for root, dirs, files in os.walk(DATASET_ROOT):
    for file in files:
        # 檢查檔案副檔名是否為影像格式
        if file.lower().endswith(IMAGE_EXTENSIONS):
            full_path = Path(root) / file
            all_images.append(full_path)

print(f"[+] 搜尋完畢！在所有子資料夾中總共找到: {len(all_images)} 張圖片")

if len(all_images) == 0:
    print("[!] 警告：找不到任何圖片，請檢查 DATASET_ROOT 路徑是否正確！")
    exit()

# =====================================================================
# 3. 隨機打散並計算劃分數量
# =====================================================================
random.seed(42)  # 固定隨機種子以確保結果可再現
random.shuffle(all_images)

total_count = len(all_images)
train_count = int(total_count * TRAIN_RATIO)

train_files = all_images[:train_count]
val_files = all_images[train_count:]

print(f"[-] 依據 8:2 比例劃分完成：")
print(f"    - 訓練集 (Train): {len(train_files)} 張")
print(f"    - 驗證集 (Val): {len(val_files)} 張")

# =====================================================================
# 4. 建立新資料夾並複製檔案（自動動態尋找標籤）
# =====================================================================
def copy_dataset_files(file_list, subset_name):
    img_out_dir = Path(OUTPUT_ROOT) / subset_name / 'images'
    lbl_out_dir = Path(OUTPUT_ROOT) / subset_name / 'labels'
    
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)
    
    for img_path in file_list:
        # 1. 複製影像檔案到新的扁平化目錄
        shutil.copy(img_path, img_out_dir / img_path.name)
        
        # 2. 動態尋找對應的標籤文字檔 (.txt)
        # 策略：不論原本 labels 與 images 怎麼擺，
        # 我們假設該圖片所屬的資料夾結構中，上層或同層有 labels 資料夾。
        lbl_name = img_path.stem + '.txt'
        
        # 嘗試尋找路徑方案 A：同層級的 labels 資料夾 (常見於一些標籤工具)
        lbl_path_a = img_path.parent / 'labels' / lbl_name
        # 嘗試尋找路徑方案 B：上一層級的 labels 資料夾 (常見於 Roboflow 格式)
        lbl_path_b = img_path.parent.parent / 'labels' / lbl_name
        # 嘗試尋找路徑方案 C：與圖片在同一個資料夾內
        lbl_path_c = img_path.parent / lbl_name
        
        # 決定最終採用的標籤路徑
        if lbl_path_a.exists():
            final_lbl_path = lbl_path_a
        elif lbl_path_b.exists():
            final_lbl_path = lbl_path_b
        elif lbl_path_c.exists():
            final_lbl_path = lbl_path_c
        else:
            final_lbl_path = None
            
        # 複製標籤
        if final_lbl_path and final_lbl_path.exists():
            shutil.copy(final_lbl_path, lbl_out_dir / lbl_name)
        else:
            # 若挖遍了子資料夾都找不到標籤，自動建立空文字檔作為背景樣本
            open(lbl_out_dir / lbl_name, 'x').close()

print("[-] 開始複製檔案並調整結構，請稍候...")
copy_dataset_files(train_files, 'train')
copy_dataset_files(val_files, 'valid')
print("[+] 所有隱藏在子資料夾的檔案已成功導出並複製完畢！")

# =====================================================================
# 5. 自動生成全新的 data.yaml
# =====================================================================
yaml_content = f"""path: {OUTPUT_ROOT} # 資料集根目錄
train: train/images
val: valid/images

nc: 9
names: ['aluminum_can', 'chip_packet', 'cork_sheet', 'one_time_foam_box', 'one_time_paper_cup', 'one_time_plastic_cup', 'one_time_plastic_plate', 'plastic_bottle', 'polythene_bag']
"""

yaml_path = Path(OUTPUT_ROOT) / 'data.yaml'
with open(yaml_path, 'w', encoding='utf-8') as f:
    f.write(yaml_content)

print(f"[+] 全新設定檔已生成至: {yaml_path}")
