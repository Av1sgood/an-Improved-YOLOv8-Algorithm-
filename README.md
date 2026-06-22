# an-Improved-YOLOv8-Algorithm-
 Recurrence the improved YOLOv8 Algorithm
#這是復現論文的readme說明
1.需要先到roboflow下載資料集
https://universe.roboflow.com/rifatx/floating-garbage-detection-aeqxx

2.下載完畢建立基於復現論文的新環境

    建立新環境名為yolov8_env
    python3 -m venv yolov8_env
    進入環境
    source yolov8_env/bin/activate
    更新套件
    pip install --upgrade pip
    下載pytorch
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128  (這邊下載128是對應RTX5080需要較新架構)
    下載物理剪枝
    pip install ultralytics torch-pruning
    測試環境是否建立完成
    python -c "import torch; import ultralytics; import torch_pruning; print('CUDA (硬體加速) 可用狀態:', torch.cuda.is_available())"

3.下載所有套件
    python3 -m venv yolov8_env && source yolov8_env/bin/activate && pip install --upgrade pip && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && pip install ultralytics torch-pruning

4.導入程式碼

5.設定資料集路徑，先執行spilt.py，將資料集拆分成train:val=8:2，會生成dataset_spilt的資料集

6.導入custom_modules程式，執行train.py開始訓練

7.訓練完畢後將模型導入測試程式

8.執行optimize_best進行物理剪枝

作者：沈英豐
學號：M11413084
班級：電子工程系碩士班
