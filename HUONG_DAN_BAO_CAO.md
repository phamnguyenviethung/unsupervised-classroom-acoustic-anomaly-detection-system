# HƯỚNG DẪN CHI TIẾT & TÀI LIỆU HỖ TRỢ VIẾT BÁO CÁO
## ĐỒ ÁN MÔN HỌC DSP501 (Digital Signal & Image Processing)
### Đề tài: Unsupervised Classroom Acoustic Anomaly Detection using DSP and AI
**Trường:** FPT University HCMC | **Kỳ:** Summer 2026

---

# MỤC LỤC
1. [Tổng Quan & Ý Nghĩa Khoa Học](#1-tổng-quan--ý-nghĩa-khoa-học)
2. [Câu Hỏi Nghiên Cứu & Giả Thuyết (RQ & Hypotheses)](#2-câu-hỏi-nghiên-cứu--giả-thuyết-rq--hypotheses)
3. [Kiến Trúc Hệ Thống & Cơ Sở Lý Thuyết DSP](#3-kiến-trúc-hệ-thống--cơ-sở-lý-thuyết-dsp)
4. [Thiết Kế Thực Nghiệm & 5 Pipeline (Ablation Study)](#4-thiết-kế-thực-nghiệm--5-pipeline-ablation-study)
5. [Nguyên Tắc Unsupervised Learning & Chống Rò Rỉ Dữ Liệu](#5-nguyên-tắc-unsupervised-learning--chống-rò-rỉ-dữ-liệu)
6. [Hướng Dẫn Sử Dụng Lệnh `run.py` & Giao Diện `app.py`](#6-hướng-dẫn-sử-dụng-lệnh-runpy--giao-diện-apppy)
7. [Cấu Trúc Báo Cáo Mẫu (Template 6 Chương Chuẩn Báo Cáo Học Thuật)](#7-cấu-trúc-báo-cáo-mẫu-template-6-chương-chuẩn-báo-cáo-học-thuật)
8. [Cách Trích Dẫn & Phân Tích Kết Quả Từ Thư Mục `results/`](#8-cách-trích-dẫn--phân-tích-kết-quả-từ-thư-mục-results)

---

## 1. TỔNG QUAN & Ý NGHĨA KHOA HỌC

### 1.1. Bối cảnh bài toán
Trong môi trường lớp học thông minh (Smart Classroom), việc giám sát an ninh và an toàn thông qua âm thanh (Acoustic Monitoring) đóng vai trò then chốt nhưng gặp nhiều thách thức:
- **Tính hiếm gặp của âm thanh bất thường:** Tiếng la hét (screaming), tiếng đập bàn/kính vỡ, tiếng bạo lực học đường hiếm khi xảy ra và rất khó thu thập đủ dữ liệu gán nhãn.
- **Tính đa dạng của âm thanh bình thường:** Tiếng giảng bài của giảng viên, tiếng thảo luận nhóm, tiếng máy chiếu, quạt trần, lật sách vở.
- **Nhiễu nền:** Âm thanh phòng học bị ảnh hưởng bởi tiếng ồn tần số thấp (hum của điều hòa, quạt gió) và tạp âm tần số cao.

### 1.2. Giải pháp đề xuất
Đề tài xây dựng một hệ thống **Học không giám sát (Unsupervised Learning)** kết hợp giữa **Xử lý tín hiệu số (DSP)** và **Mạng nơ-ron học sâu (Deep Autoencoder)**:
- Mô hình chỉ học phân phối của âm thanh lớp học bình thường (**Normal class**).
- Khi gặp âm thanh bất thường (**Abnormal class**), mô hình không thể tái tạo lại chính xác $\rightarrow$ Sai số tái tạo (Reconstruction Error) tăng vọt vượt ngưỡng $\rightarrow$ Kích hoạt cảnh báo bất thường.
- Khâu tiền xử lý DSP đóng vai trò trích xuất các đặc trưng miền tần số và loại bỏ nhiễu, giúp mạng nơ-ron hoạt động ổn định và chính xác hơn so với việc đưa trực tiếp sóng âm thô (raw audio).

---

## 2. CÂU HỎI NGHIÊN CỨU & GIẢ THUYẾT (RQ & HYPOTHESES)

Khi viết báo cáo, bạn hãy đưa 4 câu hỏi nghiên cứu và 2 giả thuyết này vào phần Mở đầu (Introduction):

- **RQ1 (DSP vs Minimal Baseline):** Tiền xử lý DSP có thực sự cải thiện độ chính xác phát hiện bất thường so với việc chỉ dùng dạng sóng thô (Baseline) trong điều kiện học không giám sát không?
- **RQ2 (Thành phần cốt lõi - Ablation):** Trong các khâu DSP (Bộ lọc thông dải Bandpass, biến đổi STFT, trích xuất năng lượng băng tần Band-Power), khâu nào đóng góp lớn nhất vào hiệu năng?
- **RQ3 (Độ bền với nhiễu - Noise Robustness):** Pipeline tích hợp đầy đủ DSP có khả năng chống chịu trước các mức nhiễu trắng nhân tạo (Additive Gaussian Noise) tốt hơn Baseline không?
- **RQ4 (Phân tích lỗi - Error Diagnostics):** Các trường hợp Báo động nhầm (False Positive) và Bỏ sót (False Negative) bắt nguồn từ những đặc trưng âm học nào?

**Giả thuyết:**
- **H1:** Pipeline `full_dsp` (Proposed) cho F1-Score, ROC-AUC và độ bền trước nhiễu vượt trội so với `baseline`.
- **H2:** Việc loại bỏ bất kỳ khối DSP nào (Bandpass, STFT, hoặc Band-Power) đều làm suy giảm đáng kể hiệu năng tổng thể hoặc tính ổn định.

---

## 3. KIẾN TRÚC HỆ THỐNG & CƠ SỞ LÝ THUYẾT DSP

### 3.1. Pipeline xử lý luồng âm thanh
```
[Raw WAV Audio (16kHz)] 
       │
       ▼
[1. Pre-emphasis & Normalization]
       │
       ▼
[2. IIR Butterworth Bandpass Filter (80 Hz - 7500 Hz)] ── (Loại bỏ hum 50/60Hz và nhiễu aliasing cao tần)
       │
       ▼
[3. Short-Time Fourier Transform (STFT)] ── (Window: Hann 1024, Hop: 256)
       │
       ▼
[4. Spectral Sub-Band Power Pooling] ── (Chia 6 dải: Sub-bass, Bass, Low-mid, Mid, Upper-mid, High)
       │
       ▼
[5. Z-score Standard Scaler (fit only on Normal)]
       │
       ▼
[6. PyTorch Deep Autoencoder (Encoder-Decoder)]
       │
       ▼
[7. Reconstruction MSE Calculation & Threshold Decision (99th Percentile)]
       │
       ▼
[Kết luận: NORMAL (Bình thường) / ANOMALY (Bất thường)]
```

### 3.2. Các công thức toán học cần đưa vào báo cáo
1. **Lọc thông dải không lệch pha (Zero-phase IIR Filtering):**
   Sử dụng hàm truyền bộ lọc Butterworth bậc 4, lọc 2 chiều qua `scipy.signal.sosfiltfilt`:
   $$|H(j\omega)|^2 = \frac{1}{1 + \left(\frac{\omega}{\omega_c}\right)^{2N}}$$
   *Ý nghĩa:* Giữ nguyên pha của tín hiệu âm thanh, tránh hiện tượng méo trễ thời gian.

2. **Biến đổi Fourier ngắn hạn (STFT):**
   $$X(m, \omega) = \sum_{n=-\infty}^{\infty} x[n] w[n - m] e^{-j\omega n}$$
   *Ý nghĩa:* Chuyển đổi tín hiệu 1D miền thời gian sang biểu diễn 2D Thời gian - Tần số (Spectrogram).

3. **Năng lượng băng tần (Sub-band Power Spectral Density):**
   Với mỗi dải tần số $[f_{low}, f_{high}]$, năng lượng tại khung thời gian $m$ được tính bằng:
   $$P_k(m) = \frac{1}{|F_k|} \sum_{f \in F_k} |X(m, f)|^2$$
   6 dải tần số sinh lý học được chia:
   - Sub-bass & Rumble: 80 - 250 Hz
   - Bass & Low Vocals: 250 - 500 Hz
   - Low Midrange (Speech formants): 500 - 1000 Hz
   - Midrange (Classroom conversation): 1000 - 2000 Hz
   - Upper Midrange (Shouting/screaming presence): 2000 - 4000 Hz
   - High Frequency (Impacts, glass breaking, crashes): 4000 - 7500 Hz

4. **Sai số tái tạo (Reconstruction Error - MSE):**
   $$L(x, \hat{x}) = \frac{1}{D} \sum_{i=1}^D (x_i - \hat{x}_i)^2$$

5. **Ngưỡng quyết định bất thường (Anomaly Threshold $\theta$):**
   $$\theta = \text{Percentile}_{99\%} \left( \{ L(x^{(j)}, \hat{x}^{(j)}) \}_{j \in \mathcal{D}_{\text{train\_normal}}} \right)$$
   Nếu $L(x_{\text{test}}, \hat{x}_{\text{test}}) > \theta \implies \text{Dự đoán = 1 (Abnormal)}$, ngược lại $\text{Dự đoán = 0 (Normal)}$.

---

## 4. THIẾT KẾ THỰC NGHIỆM & 5 PIPELINE (ABLATION STUDY)

Để chứng minh vai trò của từng tầng DSP theo đúng chuẩn nghiên cứu khoa học:

| Tên Pipeline | Tiền xử lý DSP | Biểu diễn đặc trưng | Mô hình học máy |
|---|---|---|---|
| **`full_dsp` (Proposed)** | Butterworth Bandpass (80-7500Hz) | STFT + Spectral Band-Power (30 chiều) | Autoencoder (PyTorch) |
| **`without_bandpass`** | Không lọc dải (giữ nguyên dải thô) | STFT + Spectral Band-Power (30 chiều) | Autoencoder (PyTorch) |
| **`without_bandpower`** | Butterworth Bandpass | STFT $\rightarrow$ Mean Spectrum Pooling | Autoencoder (PyTorch) |
| **`without_stft`** | Butterworth Bandpass | Time-Domain Envelope Sub-bands | Autoencoder (PyTorch) |
| **`baseline`** | Không xử lý DSP nâng cao | Downsampled Raw Waveform Block (67 chiều) | Autoencoder (PyTorch) |

---

## 5. NGUYÊN TẮC UNSUPERVISED & CHỐNG RÒ RỈ DỮ LIỆU

Báo cáo cần nhấn mạnh tính chặt chẽ trong thiết kế thực nghiệm:
1. **Zero Contamination:** Mạng nơ-ron và bộ chuẩn hóa `StandardScaler` **hoàn toàn không được nhìn thấy bất kỳ mẫu Abnormal nào** trong quá trình huấn luyện.
2. **Quy tắc chia dữ liệu:**
   - **Tập Train:** 80% dữ liệu Normal.
   - **Tập Test Đánh Giá:** 20% dữ liệu Normal còn lại + 100% dữ liệu Abnormal.
3. **Xác định ngưỡng không thiên vị:** Ngưỡng cảnh báo được tính thuần túy từ phân vị 99% của tập Train Normal, không dùng tập Test để chỉnh ngưỡng.

---

## 6. HƯỚNG DẪN SỬ DỤNG LỆNH `run.py` & GIAO DIỆN `app.py`

### 6.1. Hướng dẫn sử dụng file `run.py` (Dòng lệnh CLI)
File `run.py` là script thực nghiệm tự động toàn diện. Khi chạy, nó sẽ thực hiện từ A-Z: nạp dữ liệu $\rightarrow$ huấn luyện 5 pipeline $\rightarrow$ đánh giá $\rightarrow$ test độ bền với nhiễu $\rightarrow$ xuất toàn bộ bảng số liệu và biểu đồ vào thư mục `results/`.

#### Các lệnh cơ bản:
```bash
# 1. Chạy mặc định toàn bộ quy trình:
python run.py

# 2. Chạy với số epoch tùy chỉnh (ví dụ 30 epoch):
python run.py --epochs 30

# 3. Chỉnh ngưỡng phân vị (ví dụ 95% thay vì 99%):
python run.py --threshold-percentile 95.0

# 4. Chỉ định thư mục chứa dữ liệu âm thanh khác:
python run.py --data my_custom_audio_folder/

# 5. Cố định random seed để kết quả lặp lại chính xác 100%:
python run.py --seed 42
```

#### Bảng tham số chi tiết của `run.py`:
- `--config`: Đường dẫn file cấu hình YAML (mặc định: `config/config.yaml`).
- `--data`: Thư mục chứa audio (mặc định: `data`).
- `--sample-rate`: Tần số lấy mẫu âm thanh (mặc định: `16000` Hz).
- `--epochs`: Số lượt huấn luyện mạng PyTorch Autoencoder (mặc định: `25`).
- `--batch-size`: Kích thước batch (mặc định: `16`).
- `--lr`: Tốc độ học Learning Rate (mặc định: `0.001`).
- `--threshold-percentile`: Phân vị xác định ngưỡng bất thường (mặc định: `99.0`).
- `--seed`: Hạt giống ngẫu nhiên (mặc định: `42`).

---

### 6.2. Hướng dẫn sử dụng Giao diện Web `app.py` (Streamlit)
Chạy lệnh trong terminal:
```bash
streamlit run app.py
```
Giao diện gồm 5 tab chức năng:
1. **Tab 1: 🎙️ Dual Pipeline Live Demo**: Cho phép tải lên file WAV hoặc ghi âm trực tiếp 5 giây qua micro để so sánh kết quả phát hiện giữa **Baseline** và **Proposed Full DSP**.
2. **Tab 2: 🔄 Retrain & Experimentation**: Bấm nút huấn luyện lại toàn bộ 5 mô hình với các siêu tham số tùy biến ngay trên web.
3. **Tab 3: 📊 Quantitative Analytics**: Xem bảng số liệu so sánh, ma trận nhầm lẫn (Confusion Matrix), đường cong ROC và PR.
4. **Tab 4: 🔬 Ablation & Noise Robustness**: Xem biểu đồ so sánh đóng góp của từng khâu DSP và đồ thị đánh giá độ bền khi thêm nhiễu trắng.
5. **Tab 5: 🔍 Error Diagnostics**: Phân tích chi tiết các ca Báo động giả (False Positive) và Bỏ sót bất thường (False Negative).

---

## 7. CẤU TRÚC BÁO CÁO MẪU (TEMPLATE CHUẨN ĐỒ ÁN DSP501)

Dưới đây là dàn ý chi tiết từng chương mà bạn có thể áp dụng trực tiếp để soạn thảo file Word/PDF nộp cho giảng viên:

### CHƯƠNG 1: GIỚI THIỆU & PHÁT BIỂU BÀI TOÁN (INTRODUCTION)
- **1.1. Đặt vấn đề:** Tầm quan trọng của giám sát âm học trong lớp học thông minh.
- **1.2. Thách thức:** Dữ liệu bất thường khan hiếm $\rightarrow$ Sự cần thiết của hướng tiếp cận Unsupervised Learning.
- **1.3. Mục tiêu nghiên cứu:** Đánh giá định lượng tác động của các tầng DSP đối với chất lượng mô hình.
- **1.4. Đóng góp của đề tài:** Xây dựng hệ thống hoàn chỉnh, thực hiện Ablation Study trên 5 biến thể, kiểm thử độ bền nhiễu và cung cấp công cụ trực quan hóa thời gian thực.

### CHƯƠNG 2: CƠ SỞ LÝ THUYẾT & PHƯƠNG PHÁP (METHODOLOGY)
- **2.1. Chuỗi xử lý tín hiệu số (DSP Pipeline):**
  - Trình bày toán học về Lọc IIR Butterworth thông dải không lệch pha (80–7500 Hz).
  - Trình bày toán học về STFT (Khung cửa sổ Hann, Kích thước cửa sổ, Độ nhảy khung).
  - Trình bày phương pháp tính Năng lượng băng phổ (Spectral Sub-band Power Pooling) trên 6 dải tần.
- **2.2. Kiến trúc Mạng Nơ-ron Autoencoder:**
  - Mô tả cấu trúc Encoder - Bottleneck - Decoder trong PyTorch.
  - Hàm mất mát MSE và cơ chế phát hiện bất thường dựa trên sai số tái tạo.
- **2.3. Quy trình học không giám sát & Cơ chế tính ngưỡng 99th-percentile.**

### CHƯƠNG 3: THIẾT KẾ THỰC NGHIỆM (EXPERIMENTAL SETUP)
- **3.1. Tập dữ liệu:** Mô tả tập âm thanh lớp học Normal (giảng bài, thảo luận) và Abnormal (la hét, va đập, kính vỡ).
- **3.2. Thiết kế 5 Pipeline biến thể:** Bảng so sánh `full_dsp`, `without_bandpass`, `without_bandpower`, `without_stft`, `baseline`.
- **3.3. Các chỉ số đánh giá định lượng:**
  - Accuracy, Precision, Recall, F1-Score.
  - ROC-AUC (Area Under ROC Curve) & PR-AUC (Precision-Recall AUC).
  - False Alarm Rate (FAR) & Detection Rate (DR).
- **3.4. Kịch bản kiểm thử độ bền với nhiễu (Noise Robustness Protocol):** Thêm nhiễu Gaussian với độ lệch chuẩn $\sigma \in \{0.01, 0.03, 0.05, 0.10\}$.

### CHƯƠNG 4: KẾT QUẢ THỰC NGHIỆM & THẢO LUẬN (RESULTS & DISCUSSION)
- **4.1. Bảng số liệu tổng hợp (Trích từ `results/summary_metrics.csv`):** So sánh 5 pipeline.
- **4.2. Phân tích Ablation Study (Trả lời RQ1 & RQ2):**
  - Đánh giá sự vượt trội của `full_dsp` so với `baseline`.
  - Phân tích vai trò của bộ lọc thông dải và trích xuất đặc trưng STFT.
- **4.3. Phân tích độ bền trước nhiễu (Trả lời RQ3 - Trích từ `results/noise_robustness.png`):**
  - Biểu đồ suy giảm F1 khi tăng độ lớn của nhiễu.
  - Tại sao bộ lọc Bandpass và STFT giúp triệt tiêu nhiễu tốt hơn Baseline.
- **4.4. Phân tích lỗi & Nhận diện hạn chế (Trả lời RQ4 - Trích từ `results/error_analysis/`):**
  - Phân tích nguyên nhân các ca False Positive và False Negative.

### CHƯƠNG 5: HỆ THỐNG PHẦN MỀM & DEMO (SYSTEM IMPLEMENTATION)
- **5.1. Kiến trúc mã nguồn:** Giới thiệu các module `src/dsp.py`, `src/features.py`, `src/models.py`.
- **5.2. Giao diện Streamlit tương tác:** Hình ảnh minh họa việc nhận diện theo thời gian thực từ microphone/tệp tải lên.

### CHƯƠNG 6: KẾT LUẬN & HƯỚNG PHÁT TRIỂN (CONCLUSION)
- Tóm tắt các kết quả đạt được.
- Đề xuất mở rộng: Nhúng trên phần cứng Edge (Raspberry Pi), áp dụng kiến trúc Transformer/Conformer.

---

## 8. CÁCH TRÍCH DẪN & PHÂN TÍCH KẾT QUẢ TỪ THƯ MỤC `results/`

Sau khi bạn chạy `python run.py` (hoặc bấm Retrain trên web), hệ thống sẽ tạo sẵn các file trong `results/`. Bạn chỉ cần copy ảnh và số liệu vào báo cáo:

1. **Bảng số liệu chính:** Mở file `results/summary_metrics.csv` copy vào Chương 4.
2. **Hình vẽ đường cong ROC:** Chèn file `results/roc_curves.png` để chứng minh khả năng phân tách nhãn của mô hình.
3. **Hình vẽ PR Curves:** Chèn file `results/precision_recall_curves.png`.
4. **Ma trận nhầm lẫn:** Chèn các file `results/confusion_matrix_full_dsp.png` và `results/confusion_matrix_baseline.png` để so sánh trực quan số ca đoán đúng/sai.
5. **Biểu đồ Ablation:** Chèn `results/ablation_comparison.png` để chỉ ra đóng góp của từng khối DSP.
6. **Biểu đồ kiểm thử nhiễu:** Chèn `results/noise_robustness.png` để chứng minh mô hình hoạt động bền vững ngoài thực tế.

---
*Chúc bạn hoàn thành xuất sắc bài báo cáo môn DSP501!*
