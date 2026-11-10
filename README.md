# Phát hiện và phân loại rác thải bằng YOLOv8n

Đồ án nhóm môn Deep Learning (Đại học Bách khoa Hà Nội).
Nhóm: Đào Anh Quân, Bùi Thị Linh.

Mô hình nhận một **bức ảnh** và trả về **các khung quanh từng món rác**, kèm **loại rác** và **độ tin cậy**.
Ví dụ: "khung này là Nhựa (Plastic), tin cậy 81%".

---

## 1. Bài toán nói đơn giản

| Bài toán | Máy trả lời |
|---|---|
| Phân loại ảnh | "Ảnh này có rác nhựa" (một nhãn cho cả ảnh) |
| **Phát hiện vật thể (bài này)** | "Có 3 món rác nhựa, **ở các vị trí này**" (nhiều khung, mỗi khung một nhãn) |

Chúng tôi dùng **YOLOv8n** (bản "nano": nhỏ và nhanh nhất của YOLOv8). YOLO nhìn ảnh **một lần** rồi đưa ra tất cả khung, nên chạy nhanh.

**9 loại rác:** Animal Waste, Construction Waste, Garbage Bag, Glass, Metal, Organic, Paper, Plastic, waste (rác khác).

---

## 2. Kết quả (đọc nhanh)

Tất cả điểm nằm từ 0 đến 1, **càng cao càng tốt**. Điểm chính thức là điểm trên **tập test** (ảnh máy chưa từng thấy khi học).

| Tập | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| Valid | 0,851 | 0,656 | 0,752 | 0,579 |
| **Test** | **0,850** | **0,723** | **0,799** | **0,623** |

Cách đọc:
- **Precision 0,85**: trong các khung máy vẽ ra, khoảng 85% là đúng.
- **Recall 0,72**: trong rác thật có trong ảnh, máy tìm ra khoảng 72%.
- **mAP@0.5 0,80**: điểm tổng hợp, tính một khung là đúng khi đúng loại và chồng khớp với khung thật từ 50% trở lên.
- **mAP@0.5:0.95 0,62**: điểm khắt khe hơn (khung phải khít hơn nhiều), nên thấp hơn. Hai điểm mAP không so trực tiếp với nhau được.

**Tốc độ:** khoảng 4 ms để suy luận một ảnh (khoảng 6,7 ms cả quy trình) trên GPU Tesla T4.

### Theo từng loại (tập test)

| Lớp | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| Animal Waste | 0,851 | 0,622 | 0,717 | 0,413 |
| Construction Waste | 0,938 | 0,780 | 0,876 | 0,719 |
| Garbage Bag | 0,813 | 0,701 | 0,802 | 0,602 |
| Glass | 0,819 | 0,584 | 0,686 | 0,463 |
| Metal | 0,772 | 0,897 | 0,882 | 0,757 |
| Organic | 0,795 | 0,704 | 0,797 | 0,629 |
| Paper | 0,928 | 0,877 | 0,914 | 0,810 |
| Plastic | 0,845 | 0,866 | 0,884 | 0,727 |
| waste | 0,889 | 0,477 | 0,636 | 0,487 |

Nhận xét: loại **Paper, Plastic, Metal, Construction Waste** được nhận tốt. Loại **waste** (rác chung chung) và **Glass** (thủy tinh, trong suốt nên khó thấy) bị bỏ sót nhiều nhất (Recall thấp).

---

## 3. Trong repo này có gì

```
.
├── Phat_hien_rac_YOLOv8n_chu_thich_tieng_Viet.ipynb   <- Notebook chạy toàn bộ, có chú thích tiếng Việt
└── RAC_results/                                         <- Kết quả sau khi chạy xong trên Colab
    ├── train_run/               Quá trình học (20 epoch)
    │   ├── weights/best.pt      MÔ HÌNH TỐT NHẤT (dùng file này để dự đoán)
    │   ├── weights/last.pt      Mô hình ở epoch cuối
    │   ├── results.csv          Điểm số từng epoch
    │   ├── results.png          Đồ thị loss và mAP theo epoch
    │   ├── confusion_matrix*.png  Ma trận nhầm lẫn (lớp nào hay bị nhầm với lớp nào)
    │   ├── Box{P,R,F1,PR}_curve.png  Đường cong Precision, Recall, F1, Precision-Recall
    │   └── args.yaml, labels.jpg, train_batch*.jpg, val_batch*.jpg
    └── eval_and_predict/
        ├── val/                 Kết quả chấm trên tập VALID
        ├── val-2/               Kết quả chấm trên tập TEST (điểm chính thức)
        └── predict/             Một ảnh thử dự đoán, đã vẽ khung
```

Lưu ý: thư mục `val-2` là tập **test** (do thư viện tự đánh số khi chấm lần thứ hai).

---

## 4. Dùng thử mô hình đã học (không cần học lại)

Cài thư viện (Python 3.10 đến 3.12):

```bash
pip install ultralytics
```

Dự đoán trên một ảnh của bạn:

```python
from ultralytics import YOLO

model = YOLO("RAC_results/train_run/weights/best.pt")

# conf = ngưỡng tin cậy (chỉ giữ khung từ 25% trở lên)
# iou  = ngưỡng NMS (loại khung trùng nhau, mặc định 0.7)
ket_qua = model.predict("anh_cua_ban.jpg", conf=0.25, iou=0.7, save=True)

for khung in ket_qua[0].boxes:
    ten_lop = model.names[int(khung.cls[0])]
    tin_cay = float(khung.conf[0])
    print(ten_lop, round(tin_cay, 2), khung.xyxy[0].tolist())
```

Ảnh có vẽ khung sẽ được lưu trong thư mục `runs/detect/predict`.

---

## 5. Chạy lại toàn bộ trên Google Colab

1. Mở Colab, chọn **File > Upload notebook**, tải lên file `Phat_hien_rac_YOLOv8n_chu_thich_tieng_Viet.ipynb`.
2. Bật GPU: **Runtime > Change runtime type > T4 GPU**.
3. Tải bộ dữ liệu về máy (xem mục 7), đặt file ZIP **ở ngoài cùng của Google Drive** với tên `asian-waste-detection-challenge.zip`.
4. Chạy lần lượt các ô từ trên xuống. Mỗi ô có chú thích tiếng Việt giải thích từng dòng.
5. Huấn luyện (Bước 9) mất khoảng 45 phút trên T4.

Notebook gồm các bước: cài thư viện, lấy dữ liệu, kiểm tra dữ liệu, huấn luyện, chấm điểm trên valid, chấm điểm trên test, thử một ảnh, lưu kết quả.

---

## 6. Cách làm (tóm tắt)

- **Mô hình xuất phát:** `yolov8n.pt`, đã học sẵn trên bộ ảnh COCO (80 loại vật thể).
- **Transfer learning / fine-tune:** học tiếp **toàn bộ** mạng trên bộ ảnh rác, đổi đầu ra từ 80 lớp sang 9 lớp. Nhóm chưa thử đóng băng backbone để so sánh.
- **Cấu hình:** 20 epoch, ảnh 640x640, batch 16, dừng sớm nếu 8 epoch liền không tiến bộ, bộ tối ưu AdamW (lr khoảng 0,00077, thư viện tự chọn).
- **Data augmentation** (lật ngang, mosaic, đổi màu...): dùng mặc định của thư viện.
- **Cách chấm:** valid dùng trong lúc học để chọn `best.pt`; test chỉ dùng một lần cuối để báo cáo.

---

## 7. Dữ liệu

- **Asian Waste Detection Challenge** (Kaggle), gốc từ Roboflow, giấy phép **CC BY 4.0**.
- Train 9.964 ảnh, Valid 1.327 ảnh, Test 1.336 ảnh (tổng 12.627 ảnh), 9 lớp, nhãn dạng YOLO.
- **Dữ liệu không nằm trong repo này** (quá nặng). Cuộc thi trên Kaggle đã kết thúc, tải bộ dữ liệu về bằng tay từ trang cuộc thi.
- Một số ảnh minh họa trong `RAC_results/` (ví dụ `train_batch*.jpg`, `val_batch*.jpg`, ảnh trong `predict/`) lấy từ bộ dữ liệu này.

---

## 8. Hạn chế (nói thật)

- Mô hình chỉ học **20 epoch**; epoch tốt nhất là epoch cuối, nghĩa là điểm số có thể còn tăng nếu học lâu hơn.
- Khoảng **4 đến 5%** ảnh valid/test giống hệt ảnh trong train, nên điểm có thể hơi lạc quan.
- Chưa so sánh trực tiếp với Faster R-CNN và chưa thử trên thiết bị nhỏ (điện thoại, mạch nhúng).
- Loại `waste` và `Glass` còn bị bỏ sót nhiều.
