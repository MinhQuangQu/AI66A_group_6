# **The Accuracy Illusion in Minimal Clinical Records: Theoretical Bounds and Calibration Framework for Sepsis Survival Prediction**

**Authors:** Lê Ngọc Anh Thư, Lê Đức Minh, Nguyễn Quang Minh  
**Institution:** Faculty of DSAI, College of Technology, National Economics University (NEU)

## **Abstract**

Nghiên cứu này khảo sát các giới hạn toán học và kiến trúc của các mô hình học máy khi áp dụng vào tập dữ liệu lâm sàng tối giản (minimal clinical records) cho tiên lượng Sepsis. Chúng tôi chứng minh rằng việc quá tập trung vào chỉ số Accuracy dẫn đến một "ảo giác" về hiệu suất trong điều kiện mất cân bằng lớp nghiêm trọng. Bằng cách áp dụng lý thuyết Cận lỗi Bayes (Bayes Error Bound), nghiên cứu xác định một "Feature Bottleneck" không thể vượt qua nếu chỉ dựa trên các biến nhân khẩu học cơ bản. Chúng tôi đề xuất một khung đánh giá mới tập trung vào tính hiệu chuẩn (calibration) và các số liệu phản ánh lớp thiểu số (MCC, PR-AUC). Kết quả thực nghiệm cho thấy việc hậu hiệu chuẩn (post-hoc calibration) giúp giảm Brier Score từ 0.2246 xuống 0.0657, cung cấp một hệ thống hỗ trợ quyết định y khoa đáng tin cậy hơn so với các mô hình phức tạp nhưng thiếu hiệu chuẩn.

## **1\. Introduction**

Sepsis là một tình trạng cấp cứu lâm sàng với tỷ lệ tử vong tăng nhanh theo từng giờ chậm trễ điều trị. Tuy nhiên, việc xây dựng mô hình dự báo từ dữ liệu tối giản (tuổi, giới tính, số đợt bệnh) đối mặt với thách thức về không gian đặc trưng (feature space) hạn chế và sự mất cân bằng dữ liệu (imbalance ratio \~12.6:1). Nghiên cứu này không hướng tới việc chạy đua theo độ chính xác thuần túy mà tập trung vào việc giải mã các ranh giới lý thuyết và cải thiện chất lượng xác suất dự báo phục vụ ứng dụng lâm sàng thực tế.

## **2\. Literature Review and Theoretical Fallacies**

Các nghiên cứu tiền nhiệm thường rơi vào hai sai lầm chính:

1. **Accuracy Illusion:** Sử dụng các phương pháp lấy mẫu lại (như SMOTE) để đạt độ chính xác \>90%, nhưng thực tế chỉ là sự ghi nhớ nhiễu trong không gian đặc trưng thấp. \[1\] \[2\]  
2. **Cross-Cohort Degradation:** Các mô hình thường mất khả năng tổng quát hóa khi chuyển từ tập dữ liệu sơ cấp sang các tập dữ liệu ngoại lai (như từ Na Uy sang Hàn Quốc) do sự dịch chuyển phân phối (distribution shift). \[1\]

## **3\. Theoretical Framework and Mathematical Bounds**

### **3.1. Feature Bottleneck and Statistical Inseparability**

Dữ liệu cho thấy sự chồng lấp mật độ (density overlap) cực cao giữa hai lớp Sống/Chết. Với biến age có hệ số Cohen's d là 0.6612, tín hiệu lâm sàng là có tồn tại nhưng không đủ để phân tách tuyến tính.

### **3.2. Empirical Bayes Error Bound**

Để xác định giới hạn lý thuyết, chúng tôi sử dụng định lý Cover & Hart (1967) dựa trên sai số của thuật toán 1-Nearest Neighbor (E\_1NN). \[6\] Cận lỗi Bayes (R\*) được xác định bởi:  
`R* ≤ E_1NN ≤ 2R*(1 - R*)`  
Với E\_1NN thực nghiệm là 12.27%, chúng tôi tính toán được R\* ≈ 6.57%. Vì tỷ lệ tử vong gốc (baseline mortality) là 7.35%, khoảng cách cực hẹp này chứng minh một ranh giới toán học rằng không có mô hình nào có thể vượt qua ngưỡng sai số này một cách đáng kể nếu không bổ sung thêm các dấu hiệu sinh tồn (biomarkers).

## **4\. Methodology**

### **4.1. Feature Engineering**

Mở rộng không gian đặc trưng 3D lên 12D thông qua:

* **Polynomial Terms:** age^2, episode^2 để bắt giữ các tác động phi tuyến.  
* **Interaction Terms:** age × sex, age × episode để mô hình hóa sự phụ thuộc giữa các yếu tố nhân khẩu học.  
* **Subgroup Target Encoding:** Mã hóa xác suất tử vong theo nhóm tuổi và giới tính để trực quan hóa rủi ro nhóm.

### **4.2. Learning Strategy**

**Cost-Sensitive Learning:** Áp dụng class\_weight='balanced' để tăng hình phạt cho các lỗi dự báo sai trên lớp tử vong. \[3\]  
**Post-hoc Calibration:** Sử dụng Isotonic Regression để điều chỉnh xác suất dự báo, giải quyết hiện tượng quá tự tin (over-confidence) của các mô hình học máy. \[4\]

## **5\. Evaluation Framework**

Thay vì Accuracy, chúng tôi sử dụng bộ chỉ số dành cho dữ liệu mất cân bằng và chất lượng xác suất:

* **Matthews Correlation Coefficient (MCC):** Đánh giá mức độ tương quan tổng thể giữa dự đoán và thực tế.  
* **Brier Score:** Đo lường sai số trung bình bình phương giữa xác suất dự báo và kết quả thực tế.  
* **Expected Calibration Error (ECE):** Đo lường sự sai lệch giữa độ tin cậy của mô hình và độ chính xác thực tế trên các phân đoạn xác suất (bins).

## **6\. Results and Analysis**

### **6.1. The Calibration Breakthrough**

Kết quả cho thấy dù khả năng phân tách lớp (discrimination) chỉ tăng nhẹ, chất lượng xác suất cải thiện đột biến:

* **Brier Score:** Giảm từ 0.2246 (mô hình chưa hiệu chuẩn) xuống 0.0657 (sau hiệu chuẩn).  
* **ECE:** Giảm từ 0.3636 xuống 0.0006, cho thấy xác suất dự báo gần như khớp hoàn hảo với tần suất thực tế.

### **6.2. Cross-Cohort Performance**

Mô hình thể hiện sự sụt giảm hiệu suất trên tập kiểm thử ngoại lai (AUROC giảm từ \~0.70 xuống \~0.58), xác nhận dự đoán của Chicco & Jurman về sự nhạy cảm của các bản ghi lâm sàng tối giản đối với sự thay đổi môi trường y tế.

## **7\. Discussion**

Kết quả chứng minh rằng đối với các tập dữ liệu có "Feature Bottleneck", mục tiêu của học máy nên chuyển từ việc "phân loại đúng" sang "cung cấp xác suất đáng tin cậy". Một mô hình có độ hiệu chuẩn cao cho phép bác sĩ lâm sàng sử dụng xác suất đầu ra như một chỉ số rủi ro thực tế thay vì một nhãn nhị phân vô hồn. Điều này đặt nền móng cho các hệ thống cảnh báo sớm (Early Warning Systems) có khả năng định lượng sự không chắc chắn (Uncertainty Quantification). \[5\]

## **8\. Conclusion**

Nghiên cứu đã vạch trần các ảo giác về hiệu suất trong dự đoán Sepsis và thiết lập các cận lỗi toán học cho dữ liệu lâm sàng tối giản. Chúng tôi khẳng định rằng việc hiệu chuẩn xác suất là bước đi quan trọng nhất để đưa các mô hình AI vào ứng dụng y tế thực tiễn, nơi mà sự tin cậy của xác suất có giá trị cao hơn độ chính xác danh nghĩa.

## **References**

1. Chicco, D., & Jurman, G. (2020). Survival prediction of patients with sepsis from age, sex, and septic episode number alone. Scientific Reports, 10(1), 17156\.  
2. Carchiolo, V., & Malgeri, M. (2024). Dataset Balancing in Disease Prediction. In Proceedings of the 13th International Conference on Data Science, Technology and Applications \- Volume 1: DATA, 293-300.  
3. He, H., & Garcia, E. A. (2009). Learning from Imbalanced Data. IEEE Transactions on Knowledge and Data Engineering, 21(9), 1263-1284.  
4. Niculescu-Mizil, A., & Caruana, R. (2005). Predicting Good Probabilities with Supervised Learning. In Proceedings of the 22nd International Conference on Machine Learning, 625-632.  
5. Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic Learning in a Random World. Springer.  
6. Cover, T., & Hart, P. (1967). Nearest Neighbor Pattern Classification. IEEE Transactions on Information Theory, 13(1), 21-27.