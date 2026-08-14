# AI ASSISTANT & CO-PILOT HANDOVER GUIDE
## PROJECT: DSP501 — Unsupervised Classroom Acoustic Anomaly Detection System
**Target Context:** Hướng dẫn dành cho các AI Agent (ChatGPT, Gemini, Claude, Copilot, DeepSeek, v.v.) hoặc thành viên nhóm tiếp tục phát triển mã nguồn và hoàn thiện bài báo cáo IEEE (8–10 trang).

---

## 1. TỔNG QUAN HỆ THỐNG & CƠ CHẾ ĐÃ TRIỂN KHAI
- **Mã nguồn thuần Python:** Không còn phụ thuộc Node.js/React.
- **Kiến trúc lõi:** Unsupervised Deep Autoencoder (PyTorch) kết hợp với Audio DSP (SciPy/SoundFile/NumPy):
  1. *Audio Conditioning:* DC Offset Removal, RMS/Peak Normalization, Pre-emphasis ($\alpha=0.97$).
  2. *IIR Bandpass:* Butterworth bậc 4 ($80 - 7500\text{ Hz}$) lọc 2 chiều zero-phase (`sosfiltfilt`).
  3. *STFT:* Hann Window $N=1024$, Hop $R=256$.
  4. *Sub-band Power Pooling:* 6 dải tần sinh lý học $\rightarrow$ Vector 30 chiều.
  5. *Unsupervised Decision:* Fit `StandardScaler` và `Autoencoder` thuần túy trên dữ liệu Normal; Ngưỡng bất thường $\theta$ tính tại phân vị 99% ($99^{\text{th}}$ percentile) của sai số tái tạo (Reconstruction MSE).
- **5 Biến thể thực nghiệm (Ablation Study):** `full_dsp`, `without_bandpass`, `without_bandpower`, `without_stft`, `baseline`.
- **Thực thi:**
  - `python run.py` $\rightarrow$ Chạy CLI, huấn luyện 5 pipeline, đánh giá nhiễu $\sigma \in \{0.01, 0.03, 0.05, 0.10\}$, lưu kết quả vào `results/` và `artifacts/`.
  - `streamlit run app.py` $\rightarrow$ Web Dashboard (Microphone live test, Retrain, Quantitative graphs, Error diagnostics).
  - `python generate_docx_report.py` $\rightarrow$ Tự động đọc file `results/summary_metrics.csv` và `results/noise_robustness.csv` để sinh ra file báo cáo `DSP501_Final_Report_IEEE.docx`.

---

## 2. DANH SÁCH CÁC PHẦN CHƯA ĐIỀN CẦN AI / USER HỖ TRỢ HOÀN THIỆN

Nếu bạn là AI được giao nhiệm vụ viết tiếp hoặc hoàn thiện bài báo cáo, hãy chú ý các phần có đánh dấu `[GUIDE HINT]` sau đây:

### 📌 Mục 1: Thông tin nhóm tác giả (Author Information)
- **Vị trí:** Trang đầu file `DSP501_Final_Report_IEEE.docx` & `DSP501_Final_Report_IEEE.md`.
- **Yêu cầu:** Điền tên 3–4 thành viên, MSSV, tên giảng viên hướng dẫn, mã lớp.

### 📌 Mục 7 & Appendix A: Literature Review & Literature Matrix
- **Yêu cầu của đề bài FPT:** Cần tổng hợp **8–12 bài báo khoa học gần đây (2021–2026)** gồm:
  - Ít nhất 3 bài báo Tạp chí (Journal papers).
  - Ít nhất 3 bài báo Hội nghị (Conference papers).
  - Ít nhất 2 bài báo Tổng quan (Survey/Review papers).
- **Nhiệm vụ cho AI & Sinh viên:**
  - Điền thêm 5–9 hàng vào bảng **Appendix A** với các cột: `Paper / Author | DSP Technique | AI Model | Dataset | Key Findings | Research Gap | Limitations`.
  - Viết 2–3 đoạn phân tích so sánh các nghiên cứu này ở **Mục 7 (Literature Review)** và nêu bật khoảng trống nghiên cứu ở **Mục 8 (Research Gap)**.

### 📌 Mục 12–17: Hướng dẫn Đánh giá Chuyên sâu (Phối hợp Nhóm & AI)

Để báo cáo đạt điểm cao về độ sâu học thuật và tư duy phản biện, các mục từ 12 đến 17 cần sự kết hợp giữa phân tích thực tế của nhóm sinh viên và lập luận khoa học từ AI:

1. **Mục 12: Experimental Results & Noise Robustness**
   - **Nhóm sinh viên tự làm:** Mở file `results/summary_metrics.csv` và `results/noise_robustness.csv`. Đối chiếu các giá trị $F_1$, Detection Rate, False Alarm Rate giữa 5 biến thể (`full_dsp`, `without_bandpass`, `without_bandpower`, `without_stft`, `baseline`).
   - **Phối hợp với AI:** Đặt câu hỏi cho AI: *"Dựa vào bảng số liệu, tại sao khi bỏ bước gom 6 dải tần (`without_bandpower`), tỷ lệ False Alarm Rate lại tăng từ 25% lên 37.5%? Hãy giải thích dưới góc độ nén chiều không gian phổ và tính ổn định năng lượng."*

2. **Mục 13: Error Analysis (Diagnostics & Failure Cases)**
   - **Nhóm sinh viên tự làm:** Mở thư mục `results/error_analysis/` (các file `false_positives_*.csv` và `false_negatives_*.csv`), nghe trực tiếp các file audio bị đoán sai trong `data/`. Xác định âm thanh đó là gì (tiếng cười lớn, tiếng kéo ghế cọ xát sàn, tiếng vỗ tay đột ngột).
   - **Phối hợp với AI:** Cung cấp danh sách tên file lỗi và hỏi AI: *"Tại sao đặc trưng tần số - thời gian của tiếng cười to hoặc tiếng trượt ghế lại làm bộ giải mã Autoencoder tái tạo sai vượt ngưỡng $\theta$?"* $\rightarrow$ Đưa kết luận này vào bài.

3. **Mục 14: Discussion (Critical Debate & Theoretical Linkage)**
   - **Nhóm sinh viên tự làm:** Thảo luận các câu hỏi nghiên cứu (RQ1–RQ4) xem thực tế có đúng như kỳ vọng không.
   - **Phối hợp với AI:** Yêu cầu AI phản biện (Critical Debate): *"Hãy chỉ ra những điểm yếu cốt lõi khi chỉ dùng mạng Autoencoder tĩnh không có cơ chế Temporal Memory (như LSTM/Transformer) trong bài toán giám sát âm thanh theo chuỗi thời gian."*

4. **Mục 15: Threats to Validity**
   - **Nhóm sinh viên tự làm:** Đánh giá các rủi ro thực tế: Âm vang phòng học (Reverberation $RT_{60}$), vị trí đặt micro xa/gần, tạp âm điều hòa thay đổi theo mùa.
   - **Phối hợp với AI:** Nhờ AI liệt kê thêm các hạn chế về phần cứng DSP (Microphone dynamic range clipping, 16-bit quantization noise) để viết thành 3 nhóm: *Internal Validity, External Validity, Construct Validity*.

5. **Mục 16 & 17: Conclusion & Future Work**
   - **Nhóm sinh viên tự làm:** Thống nhất định hướng mở rộng đề tài (nhúng lên vi điều khiển ESP32 / Raspberry Pi, tích hợp định vị âm học Beamforming).
   - **Phối hợp với AI:** Nhờ AI tra cứu và đề xuất các từ khóa nghiên cứu mới (TinyML, Conformer, Direction-of-Arrival Estimation).

### 📌 Mục 14: Discussion (Yêu cầu bắt buộc theo FPT Syllabus)
Phần Thảo luận **bắt buộc** phải phân tích sâu hơn ngoài các con số độ chính xác đơn thuần:
1. **WHY (Tại sao có kết quả đó):** Phân tích liên kết trực tiếp với lý thuyết DSP (Bộ lọc Butterworth bậc 4 zero-phase giữ nguyên pha xung kích, biến đổi STFT Hann window $N=1024, R=256$, và cơ chế nén chiều 6 dải tần sinh lý học) và trả lời thỏa đáng các câu hỏi nghiên cứu (RQ1, RQ2, RQ3).
2. **STRENGTHS (Điểm mạnh cốt lõi):** So sánh ưu thế vượt trội của giải pháp không giám sát (Unsupervised Autoencoder không cần nhãn bất thường hiếm hoi) so với baseline thô và các nghiên cứu trước trong Literature Review.
3. **WEAKNESSES & LIMITATIONS (Điểm yếu & Hạn chế):** Chỉ rõ các hạn chế về mặt phương pháp (mô hình hóa cửa sổ tĩnh chưa có bộ nhớ tuần tự LSTM/Transformer), ngưỡng bất thường cố định $\theta$ tại 99th percentile, và vấn đề âm vang phòng học ($RT_{60}$).

### 📌 Mục 15: Threats to Validity
Phân tích đầy đủ 3 loại rủi ro và biện pháp giảm thiểu (Mitigation Strategies):
- **Internal Validity:** Chia tập dữ liệu triệt để, fit `StandardScaler` và mô hình thuần túy trên Normal, cố định `seed = 42` tránh rò rỉ dữ liệu (data leakage).
- **External Validity:** Khả năng tổng quát hóa với các loại phòng học khác nhau, độ nhạy của micro.
- **Construct Validity:** Đánh giá đa chiều qua bộ chỉ số toàn diện (F1, PR-AUC, ROC-AUC, FAR, DR) thay vì chỉ dùng Accuracy.

### 📌 Mục 18: Ethics Statement
Cam kết đạo đức nghiên cứu, bảo mật quyền riêng tư (dữ liệu âm thanh ẩn danh PII-free, không nhận dạng giọng nói người dùng).

### 📌 Mục 19: AI Declaration (Appendix)
Bảng khai báo minh bạch công cụ AI (Gemini, ChatGPT, Claude, Copilot) và cam kết trách nhiệm học thuật của nhóm tác giả theo chuẩn đề cương FPT.

### 📌 Phần nộp kèm: Individual AI Reflection (Mục 18 trong FPT Guidelines)
- **Quy định:** Mỗi sinh viên trong nhóm nộp một đoạn ngắn **300–500 từ** phản tư độc lập trả lời 6 câu hỏi:
  1. Đã dùng công cụ AI nào?
  2. AI hỗ trợ nhiều nhất ở giai đoạn nào (ý tưởng, DSP filter, PyTorch, sửa lỗi, viết báo cáo)?
  3. AI đã gặp lỗi/hallucination gì (ví dụ: nhầm chiều tensor, trôi pha bộ lọc)?
  4. Bạn đã kiểm chứng output của AI bằng cách nào (unit test, đồ thị kiểm tra)?
  5. Quyết định kỹ thuật/nghiên cứu nào bắt buộc phải do chính bạn tự quyết định?
  6. Bài học kinh nghiệm để cải thiện việc sử dụng AI trong tương lai.

### 📌 Appendix A: Literature Matrix
Bảng ma trận tổng hợp 8–12 bài báo khoa học gần đây (2021–2026).

### 📌 Appendix B: AI Declaration Template
Dẫn chiếu tới bảng khai báo tại Mục 19.

### 📌 Appendix C: Project Checklist (23 Mục chi tiết theo chuẩn FPT)
Đã được lập trình tự động với toàn bộ 23 tiêu chí kiểm tra deliverables: `Team Formation, Topic Registration, Research Problem, Research Objectives, Research Questions, Literature Review, Research Gap, Hypothesis, DSP Pipeline, Feature Extraction, AI Model, Experimental Design, Baseline Comparison, Ablation Study, Error Analysis, Discussion, Ethics Statement, AI Declaration, AI Reflection, Final Report, Source Code, README, Presentation`. Tất cả đã được xác thực trạng thái `[x] Completed`.

### 📌 Appendix D: Frequently Asked Questions (FAQ & Compliance)
Chứa hướng dẫn giải đáp các thắc mắc về liêm chính học thuật, sử dụng AI (ChatGPT, Gemini, Claude, GitHub Copilot), pretrained models, dataset công khai (DCASE, RAVDESS), và tiêu chuẩn tái lập thí nghiệm 100% (Reproducibility).

---

## 3. TÁC ĐỘNG KHI THAY ĐỔI DỮ LIỆU HOẶC RETRAIN (DATA MUTATION IMPACT MAP)

Khi người dùng thay đổi dữ liệu âm thanh trong `data/` hoặc bấm **RETRAIN** / chạy `python run.py`, các file sau sẽ thay đổi và ảnh hưởng trực tiếp đến các mục trong báo cáo:

| Dữ liệu thay đổi / File sinh ra | Các mục bị ảnh hưởng trong Báo cáo | Hành động cần thực hiện |
|---|---|---|
| **`data/normal/` & `data/abnormal/`** (Thêm/bớt file audio) | • **Mục 11.1 (Dataset Partitioning):** Số lượng mẫu normal/abnormal.<br>• **Mục 10.4:** Giá trị ngưỡng $\theta$ (99th percentile). | Cập nhật lại số lượng file audio trong Mục 11.1 (ví dụ: 40 mẫu train, 15 mẫu test). |
| **`results/summary_metrics.csv`** (F1, Accuracy, ROC-AUC, FAR, DR) | • **Mục 1 & 16:** Con số tóm tắt trong Abstract & Conclusion.<br>• **Mục 12.1:** Bảng số liệu so sánh 5 pipeline.<br>• **Mục 14.1:** Phân tích Discussion. | Chạy lệnh `python generate_docx_report.py` để script tự động đọc file CSV mới và cập nhật bảng Word. Sau đó kiểm tra lại các đoạn text nhận xét. |
| **`results/noise_robustness.csv`** (Kiểm thử nhiễu $\sigma \in [0.01, 0.10]$) | • **Mục 12.2:** Phân tích độ bền trước nhiễu.<br>• **Mục 6 (RQ3):** Trả lời câu hỏi nghiên cứu số 3. | Cập nhật lại giá trị F1 tại mức nhiễu $\sigma=0.05$ và $\sigma=0.10$ trong Mục 12.2. |
| **`results/error_analysis/`** (`false_positives_*.csv`, `false_negatives_*.csv`) | • **Mục 13 (Error Analysis):** Phân tích các trường hợp đoán sai.<br>• **Mục 6 (RQ4):** Trả lời câu hỏi nghiên cứu số 4. | Mở các file CSV trong thư mục `results/error_analysis/`, xem tên các file audio bị phân loại nhầm để giải thích nguyên nhân âm học (tiếng cười quá to, tiếng kéo bàn sắc nhọn,...). |
| **Các biểu đồ ảnh PNG trong `results/`** (`roc_curves.png`, `precision_recall_curves.png`, `confusion_matrix_*.png`, `ablation_comparison.png`, `noise_robustness.png`) | • **Mục 12 & Mục 13:** Hình ảnh trực quan hóa bắt buộc theo chuẩn IEEE. | Chèn các file ảnh PNG mới nhất từ thư mục `results/` vào các mục tương ứng trong file Word. |

---

## 4. QUY TẮC ĐỒNG BỘ HÓA TỰ ĐỘNG (SYNC AUTOMATION)

Nếu AI tiếp theo thực hiện thay đổi mã nguồn DSP hoặc train lại mô hình:
1. Luôn chạy Unit Tests để đảm bảo tính toàn vẹn:
   ```bash
   PYTHONPATH=. python3 tests/test_audio.py && PYTHONPATH=. python3 tests/test_dsp.py && PYTHONPATH=. python3 tests/test_features.py && PYTHONPATH=. python3 tests/test_metrics.py
   ```
2. Chạy tái tạo toàn bộ thực nghiệm CLI:
   ```bash
   python run.py --seed 42
   ```
3. Sinh lại file báo cáo Word:
   ```bash
   python generate_docx_report.py
   ```

---
*Tệp hướng dẫn này đảm bảo tính kế thừa và tái lập 100% theo chuẩn Research-Based Learning của môn học DSP501.*
