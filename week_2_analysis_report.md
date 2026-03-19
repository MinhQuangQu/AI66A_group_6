# Báo Cáo Phân Tích Thông Tin & Đánh Giá Giới Hạn Mô Hình (Tuần 2)

Dựa trên kết quả triển khai toàn diện và các kết quả tính toán trực tiếp từ `week_2.ipynb`, dưới đây là phân tích chi tiết về mức độ tín hiệu của dữ liệu, giới hạn toán học và hiệu chuẩn chéo:

## 1. Phân Tích Thống Kê & Hiệu Ứng (Effect Size)
Theo kết quả từ Formal Statistical Tests trên các biến của tập chuẩn (Primary Cohort, $N=110,204$):
*   **Chi-Square (Sex vs Outcome):**
    *   $\chi^2$ Stat = `43.0373`, $p$-value = `5.3707e-11`.
    *   **Kết luận:** Có sự phụ thuộc có ý nghĩa thống kê giữa giới tính và tỷ lệ tử vong, tuy nhiên sự phụ thuộc này sẽ cần kết hợp mô hình học máy để trích xuất phi tuyến.
*   **Kruskal-Wallis (Age & Episode):**
    *   **Age:** H-stat = `3766.4964`, $p$-value $\approx 0$.
    *   **Episode:** H-stat = `18.4735`, $p$-value = `1.7229e-05`.
    *   **Kết luận:** Phân phối của người sống và người chết khác biệt rất rõ rệt ở độ tuổi, xác nhận rằng tuổi tác chứa tín hiệu mạnh về sinh tồn bệnh nhân.
*   **Cohen's d cho `age`:**
    *   Giá trị Cohen's d: **`0.6612`** (dấu tuyệt đối).
    *   *Insight:* Với Effect size nằm trong khoảng Khá đến Lớn (Moderate to Large, $> 0.5$), chúng ta xác nhận được tín hiệu lâm sàng quan trọng: Sự chênh lệch sống/chết do nguyên nhân từ độ tuổi không phải là ngẫu nhiên. Biến `age` sẽ là trụ cột của bất kỳ mô hình dự đoán nào.

## 2. Định Giá Sai Số Bất Khả Kháng (Empirical Bayes Error Bound)
Để tìm ra cận dưới lỗi của thuật toán học máy, thuật toán 1-Nearest Neighbor (1-NN) của Cover & Hart đã được chạy trên 20% dữ liệu phân tầng chuẩn hóa:
*   Lỗi phân loại của bộ 1-NN ($E_{1NN}$): **`12.27%`**
*   Giải phương trình bậc 2 giới hạn, thu được **Bayes Error Bound ($R^*$): `~6.57%`**
*   **Phân tích Rủi ro (Feature Bottleneck):**
    *   Tỷ lệ tử vong gốc (Baseline Mortality Rate) của toàn tập Primary là **`7.35%`**.
    *   *Insight Toán học:* Cận dưới thất bại của mô hình học máy (`6.57%`) tiệm cận cực kỳ khít với ngưỡng mô hình Dummy đoán "Tất cả đều Sống" (`7.35%`). Điều này kết luận một **Feature Bottleneck trầm trọng**: Chỉ với 3 biến `age, sex, episode` và không có các đặc trưng sinh tồn lâm sàng (như nhịp tim, huyết áp, lab tests,...), chúng ta *sẽ không thể vượt qua ngưỡng chính xác này một cách thực sự (có PR-AUC/MCC tốt)* bằng bất kỳ mô hình thuần túy nào. Models sẽ chỉ đạt mức trần dự đoán ~92.6% (vốn có do imbalanced class).

## 3. Baseline & Hệ Quả Mất Cân Bằng (Accuracy Illusion)
Xác thực lại quan điểm Mất Cân Bằng ở Kế hoạch Tuần 1:
*   Mô hình Dummy (`most_frequent` - Luôn đoán sống) mang lại Accuracy cực cao ở mức **`92.65%`**.
*   Tuy nhiên, các metric phản ánh thực tế (discrimination metrics) đều chạm đáy:
    *   **MCC (Matthews Correlation Coefficient): `0.000`**
    *   **PR-AUC: `0.926`** (do Positive Class là đa số). Tuy nhiên, AUROC chỉ là `0.500`. Brier score cực mỏng là `0.0735`.
*   *Insight:* Các paper thông báo mức Accuracy ~85-90% trên Sepsis với số lượng feature ít ỏi rõ ràng rơi vào rập khuôn "Ảo giác Accuracy" của Imbalanced data. Các chỉ số PR-AUC trên lớp hiếm (Dead) và MCC mới là bộ đo chân thực.

## 4. Stratified CV tracking & Logistic Out-of-Fold (OOF) Prediction
Thay vì sử dụng cross-val để lấy điểm, một pipeline lưu trữ xác suất Out-Of-Fold hoàn chỉnh đã được xây dựng:
*   File output quan trọng: `oof_probabilities_lr.csv` (Đã lưu vào Workspace).
*   **Phát hiện hiện tượng miscalibration:** Khi chạy dummy Logistic Regression với tham số `class_weight='balanced'` để đối phó Imbalance, Brier Score lập tức **tăng vọt lên `0.2246`** (tệ hơn nhiều lần so với mức `0.0735` của Baseline mù!).
*   *Hành động chuẩn bị cho Tuần sau:* Kết quả chênh lệch Brier Score này là nguyên liệu hoàn hảo để chúng ta thực hiện vẽ **Reliability Diagram (Calibration Curve) toàn cục**. Mô hình bị over-confident (vống xác suất) vào nhóm thiểu số `class_weight`, đòi hỏi kỹ thuật hậu hiệu chuẩn (Platt Scaling hoặc Isotonic Regression) trong những tuần tới.

---
**Tổng kết:**
Việc thiết kế luồng quy định khắt khe bằng Effect Size & Lý thuyết 1-NN đã chứng minh đầy đủ: Khả năng học sâu của thuật toán trên không gian 3 biến bị gò bó bởi cận dưới Bayes. Việc triển khai học máy ở chặng sau bắt buộc phải tập trung đi sâu vào các độ đo đánh giá lớp hiếm (Calibration, MCC, PR-AUC), thay vì cạnh tranh trên thang điểm Accuracy giả mạo.