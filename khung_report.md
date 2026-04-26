# **The Accuracy Illusion in Minimal Clinical Records: Theoretical Bounds and Calibration Framework for Sepsis Survival Prediction**

**Authors:** Lê Ngọc Anh Thư, Lê Đức Minh, Nguyễn Quang Minh  
**Institution:** Faculty of DSAI, College of Technology, National Economics University (NEU)

## **Abstract**

Nghiên cứu này khảo sát các giới hạn toán học và kiến trúc của các mô hình học máy khi áp dụng vào tập dữ liệu lâm sàng tối giản (minimal clinical records) cho tiên lượng Sepsis. Chúng tôi chứng minh rằng việc quá tập trung vào chỉ số Accuracy dẫn đến một "ảo giác" về hiệu suất trong điều kiện mất cân bằng lớp nghiêm trọng. Bằng cách áp dụng lý thuyết Cận lỗi Bayes (Bayes Error Bound), nghiên cứu xác định một "Feature Bottleneck" không thể vượt qua nếu chỉ dựa trên các biến nhân khẩu học cơ bản. Chúng tôi đề xuất một khung đánh giá mới tập trung vào tính hiệu chuẩn (calibration) và các số liệu phản ánh lớp thiểu số (MCC, PR-AUC). Kết quả thực nghiệm cho thấy việc hậu hiệu chuẩn (post-hoc calibration) giúp giảm Brier Score từ 0.2246 xuống 0.0657, cung cấp một hệ thống hỗ trợ quyết định y khoa đáng tin cậy hơn so với các mô hình phức tạp nhưng thiếu hiệu chuẩn.

## **1\. Introduction**

Sepsis là một tình trạng cấp cứu lâm sàng với tỷ lệ tử vong tăng nhanh theo từng giờ chậm trễ điều trị, đặt ra nhu cầu cấp thiết cho các hệ thống dự báo sớm đáng tin cậy. Tuy nhiên, việc xây dựng mô hình dự báo từ dữ liệu tối giản (tuổi, giới tính, số đợt bệnh) đối mặt với thách thức về không gian đặc trưng (feature space) hạn chế và sự mất cân bằng dữ liệu (imbalance ratio \~12.6:1).

Trong nghiên cứu này, chúng tôi khai thác ba cohort độc lập thay vì một tập dữ liệu đơn lẻ, bao gồm: primary cohort (110,204 lượt nhập viện) dùng để trích xuất và sàng lọc dữ liệu ban đầu, study cohort (19,051 bệnh nhân) phục vụ huấn luyện và đánh giá nội bộ, và external validation cohort (137 bệnh nhân tại Hàn Quốc) nhằm kiểm định khả năng tổng quát hóa xuyên miền. Thiết kế đa-cohort này cho phép đánh giá rõ ràng hiện tượng suy giảm hiệu suất (cross-cohort degradation) dưới tác động của distribution shift.

Dữ liệu thể hiện mức độ mất cân bằng lớp nghiêm trọng với tỷ lệ sống sót 92.65% và tử vong 7.35%, tương ứng với imbalance ratio xấp xỉ 12.6:1. Quan trọng hơn, tập dữ liệu không chứa giá trị thiếu (missing values = None), loại bỏ hoàn toàn nhu cầu về các kỹ thuật tiền xử lý như imputation hay data cleaning phức tạp. Điều này tạo điều kiện để cô lập và phân tích trực tiếp một vấn đề cốt lõi hơn: giới hạn thông tin nội tại của không gian đặc trưng (feature bottleneck) khi chỉ sử dụng các biến nhân khẩu học tối giản.

Một quyết định thiết kế quan trọng trong nghiên cứu là đảo chiều biến mục tiêu (target flipping), trong đó nhãn được chuyển từ alive = 1 sang dead = 1. Việc tái định nghĩa này không mang tính hình thức mà nhằm đảm bảo rằng các chỉ số đánh giá dành cho lớp thiểu số (như MCC, PR-AUC) phản ánh trực tiếp hiệu suất dự báo tử vong thay vì bị “che khuất” bởi lớp chiếm ưu thế là sống sót. Qua đó, nghiên cứu tránh được "ảo giác độ chính xác" (accuracy illusion) và tập trung vào giá trị lâm sàng thực sự của mô hình.

Trong bối cảnh đó, nghiên cứu này không hướng tới việc tối đa hóa độ chính xác danh nghĩa thuần túy, mà tập trung vào việc giải mã các ranh giới lý thuyết của bài toán và cải thiện chất lượng xác suất dự báo, hướng đến các hệ thống hỗ trợ quyết định y khoa có khả năng hiệu chuẩn cao và đáng tin cậy trong thực tế.

## **2\. Literature Review and Theoretical Fallacies**

Các nghiên cứu tiền nhiệm thường rơi vào hai sai lầm chính, xuất phát từ cả hạn chế phương pháp luận lẫn cách diễn giải kết quả:

### 2.1. **Accuracy Illusion:** 

Một xu hướng phổ biến trong dữ liệu y khoa có không gian đặc trưng thấp (Low-Dimensional) là sử dụng các kỹ thuật cân bằng dữ liệu như SMOTE nhằm cải thiện độ chính xác tổng thể (Accuracy). Cụ thể, các nghiên cứu tiền nhiệm đã sử dụng các phương pháp lấy mẫu lại (như SMOTE) để đạt độ chính xác \>90% \[1\]. Tuy nhiên, theo Carchiolo & Malgeri (2024), việc áp dụng SMOTE trên các tập dữ liệu có không gian đặc trưng thấp có thể dẫn đến ảo giác Accuracy - độ chính xác cao (Accuracy = 0.982) nhưng không phản ánh năng lực dự báo thực sự. Nguyên nhân cốt lõi nằm ở việc các điểm dữ liệu tổng hợp không bổ sung thông tin mới mà chỉ khuếch đại cấu trúc sẵn có của lớp đa số, khiến mô hình học cách ghi nhớ (memorize) thay vì khái quát hóa (generalize) \[2\].

Quan sát này đặc biệt quan trọng trong bối cảnh dữ liệu lâm sàng tối giản, nơi số chiều đặc trưng bị giới hạn nghiêm trọng. Khi đó, các kỹ thuật oversampling không mở rộng được biên phân tách (decision boundary) một cách có ý nghĩa, mà chỉ làm tăng mật độ điểm trong cùng một vùng chồng lấp (overlap region). Do đó, độ chính xác cao đạt được trong các thiết lập này cần được xem là một sản phẩm của phương pháp, thay vì bằng chứng của hiệu suất mô hình.

### 2.2. **Cross-Cohort Degradation:** 

Các mô hình thường mất khả năng tổng quát hóa khi chuyển từ tập dữ liệu sơ cấp sang các tập dữ liệu ngoại lai (như từ Na Uy sang Hàn Quốc) do sự dịch chuyển phân phối (distribution shift). Chicco & Jurman (2020) đã cảnh báo rằng các mô hình xây dựng trên dữ liệu tối giản rất nhạy cảm với distribution shift giữa các hệ thống y tế khác nhau. \[1\]

Trong nghiên cứu này, chúng tôi không chỉ kế thừa lập luận đó mà còn xác nhận bằng bằng chứng thực nghiệm trực tiếp. Cụ thể, mô hình đạt AUROC nội bộ ~0.706, nhưng giảm xuống còn ~0.589 trên study cohort độc lập và tiếp tục giảm nhẹ xuống ~0.572 trên validation cohort từ Hàn Quốc. Sự suy giảm nhất quán này cho thấy rằng tín hiệu học được từ các biến nhân khẩu học mang tính ngữ cảnh địa phương (context-dependent) và không đủ mạnh để duy trì hiệu suất trong môi trường ngoại lai.

Điều này củng cố giả thuyết rằng với các “minimal clinical records”, vấn đề không chỉ là overfitting theo nghĩa truyền thống, mà là giới hạn thông tin cấu trúc (structural information limit) của dữ liệu.

### 2.3 **From Literature Survey to Implementation Decisions**

Khác với nhiều báo cáo tổng quan mang tính mô tả, nghiên cứu này thực hiện một phân tích có hệ thống trên 6 công trình nền tảng, kết hợp giữa lý thuyết học máy, dữ liệu mất cân bằng, và hiệu chuẩn xác suất. Mỗi công trình không chỉ được trích dẫn mà còn trực tiếp ảnh hưởng đến quyết định thiết kế mô hình:

- Chicco & Jurman (2020): Nhấn mạnh vai trò của MCC thay cho accuracy \[1\] → dẫn đến việc lựa chọn MCC làm chỉ số đánh giá trung tâm. 
- Carchiolo & Malgeri (2024): Cảnh báo về oversampling \[2\] → loại bỏ SMOTE khỏi pipeline, chuyển sang cost-sensitive learning.
- He & Garcia (2009): Đặt nền tảng cho học với dữ liệu mất cân bằng \[3\] → triển khai class_weight='balanced'.
- Niculescu-Mizil & Caruana (2005): Chỉ ra vấn đề xác suất không được hiệu chuẩn \[4\]→ áp dụng isotonic regression cho post-hoc calibration. 
- Vovk et al. (2005): Định hướng về uncertainty quantification \[5\] → tích hợp đánh giá ECE và Brier Score.
- Cover & Hart (1967): Cung cấp cơ sở cho cận lỗi Bayes \[6\] → sử dụng 1-NN để ước lượng giới hạn lý thuyết.

Cách tiếp cận này biến phần Literature Review từ một tổng hợp thụ động thành một framework định hướng quyết định, trong đó mỗi lựa chọn kỹ thuật đều có cơ sở học thuật rõ ràng. Điều này đặc biệt quan trọng trong bối cảnh bài toán có feature bottleneck, khi cải tiến hiệu suất không thể dựa vào việc tăng độ phức tạp mô hình, mà phải đến từ hiểu đúng giới hạn của dữ liệu và lựa chọn đúng mục tiêu tối ưu hóa.

## **3\. Theoretical Framework and Mathematical Bounds**

### **3.1. Feature Bottleneck and Statistical Inseparability**

Một trong những đóng góp trung tâm của nghiên cứu là việc định lượng rõ ràng mức độ không thể phân tách thống kê (statistical inseparability) giữa hai lớp sống và tử vong trong không gian đặc trưng tối giản. Phân tích phân phối cho thấy sự chồng lấp mật độ (density overlap) đáng kể, được xác nhận thông qua *Hellinger distance* giữa hai lớp: 0.28 đối với biến age và 0.24 đối với biến episode. Các giá trị này nằm trong vùng trung gian, cho thấy tồn tại tín hiệu phân biệt nhưng không đủ mạnh để tạo ra một biên quyết định rõ ràng.

Kết quả này được củng cố bởi phép đo Mutual Information (MI) thực hiện trong giai đoạn phân tích ban đầu (Week 1), cho thấy lượng thông tin mà các đặc trưng cung cấp về biến mục tiêu là rất thấp. Nói cách khác, các biến đầu vào mang tính dự báo yếu, dẫn đến một không gian đặc trưng có tính chồng lấp cao và khó khai thác bằng các mô hình học máy truyền thống.

Để minh họa giới hạn này từ một góc nhìn thực nghiệm, nhóm chúng tôi đã thiết lập một dummy baseline (most-frequent classifier) như một cận dưới (lower bound). Mô hình này đạt Accuracy = 92.65%, tương đương với tỷ lệ lớp đa số, nhưng đồng thời cho MCC = 0.0000 và AUROC = 0.5000, phản ánh hoàn toàn sự thiếu vắng khả năng phân biệt. Kết quả này cung cấp một minh chứng trực tiếp rằng trong bối cảnh mất cân bằng mạnh và đặc trưng yếu, accuracy có thể đạt mức rất cao mà không mang bất kỳ giá trị dự báo nào, qua đó càng củng cố thêm luận điểm về “accuracy illusion”

### **3.2. Empirical Bayes Error Bound**

Để xác định giới hạn lý thuyết và giới hạn hiệu suất tối đa có thể đạt được, nghiên cứu sử dụng định lý kinh điển của Cover & Hart (1967), liên hệ giữa sai số của thuật toán 1-Nearest Neighbor (E\_1NN) và \[6\] Cận lỗi Bayes tối ưu (R\*):

`R* ≤ E_1NN ≤ 2R*(1 - R*)`  

Với giá trị thực nghiệm E_1NN = 12.27%, chúng tôi suy ra R* ≈ 6.57%. Khi so sánh với tỷ lệ tử vong nền (7.35%), có thể thấy khoảng cách giữa mô hình tối ưu lý thuyết và baseline thực tế là cực kỳ nhỏ. Điều này dẫn đến một kết luận quan trọng: không tồn tại dư địa đáng kể để cải thiện sai số phân loại nếu không bổ sung thêm thông tin đặc trưng mới (biomarkers hoặc tín hiệu sinh tồn)

Khi kết hợp các bằng chứng: (i) Hellinger distance ở mức trung bình; (ii) Mutual Information thấp; (iii) dummy baseline đạt accuracy cao nhưng vô nghĩa; và (iv) cận lỗi Bayes gần sát baseline, ta có thể khẳng định rằng bài toán này bị chi phối bởi một “feature bottleneck” mang tính cấu trúc. Giới hạn này không thể bị phá vỡ thông qua việc tăng độ phức tạp mô hình hay tối ưu hóa thuật toán, mà chỉ có thể được giải quyết bằng cách mở rộng không gian thông tin đầu vào.

Do đó, phần còn lại của nghiên cứu không theo đuổi mục tiêu vượt qua các cận này, mà chuyển hướng sang một mục tiêu thực tế hơn: tối ưu hóa chất lượng xác suất dự báo (probabilistic calibration) trong phạm vi giới hạn lý thuyết đã được xác lập.

## **4\. Methodology**

### **4.1. Feature Engineering**

Mặc dù báo cáo ban đầu mô tả việc mở rộng không gian đặc trưng từ 3D lên 12D, pipeline thực tế được triển khai chi tiết hơn đáng kể, bao gồm 7 nhóm đặc trưng (feature groups) nhằm khai thác tối đa tín hiệu hạn chế từ dữ liệu nhân khẩu học:

* **Polynomial Terms:** Bổ sung các thành phần phi tuyến như age^2, episode^2 để nắm bắt các hiệu ứng tăng tốc (non-linear escalation), đặc biệt trong các nhóm tuổi cao hoặc bệnh nhân có nhiều đợt nhiễm trùng. 
* **Interaction Terms:** Mô hình hóa sự phụ thuộc giữa các biến thông qua các tích chéo age × sex, age × episode, sex × episode, và age × sex × episode. Các tương tác bậc cao này cho phép biểu diễn các hiệu ứng điều kiện (conditional effects) mà mô hình tuyến tính thuần túy không thể nắm bắt.
* **Recurrent Indicator:** Biến nhị phân `is_recurrent` đánh dấu các trường hợp sepsis tái phát, cung cấp một tín hiệu lâm sàng quan trọng liên quan đến mức độ nghiêm trọng và nguy cơ tử vong.
* **Age Binning (One-hot Encoding):** Phân nhóm tuổi thành 5 khoảng rời rạc (0–18, 19–40, 41–60, 61–80, 81+) và mã hóa one-hot nhằm giảm nhiễu và làm nổi bật các ngưỡng rủi ro lâm sàng.
* **Subgroup Target Encoding:** Xây dựng đặc trưng xác suất có điều kiện $P(Dead | age_bin × sex)$. Quan trọng, encoding này được học chỉ trên các training folds trong cross-validation, đảm bảo không xảy ra data leakage và giữ tính hợp lệ thống kê của mô hình.

Tổng thể, các kỹ thuật này chuyển đổi không gian đặc trưng từ 3 biến gốc lên 12 biến có cấu trúc, không nhằm “tăng số lượng” một cách cơ học mà nhằm tái biểu diễn thông tin trong điều kiện feature bottleneck.

### **4.2. Learning Strategy**

Thay vì phụ thuộc vào một mô hình đơn lẻ, nghiên cứu triển khai và so sánh nhiều cấu hình học khác nhau, phản ánh các lựa chọn thiết kế có chủ ý:

* **Model Configurations:**
    * **Logistic Regression (LR – raw):** baseline tuyến tính trên đặc trưng gốc
    * **Logistic Regression (LR – engineered):** tích hợp toàn bộ feature engineering
    * **Logistic Regression (LR – calibrated):** bổ sung bước hiệu chuẩn xác suất
    * **Random Forest (RF) và XGBoost (XGB):** kiểm tra liệu mô hình phi tuyến có vượt qua được feature bottleneck hay không
* **Cost-Sensitive Learning:** Áp dụng class\_weight='balanced' để tăng trọng số cho lớp tử vong, phù hợp với khuyến nghị từ He & Garcia (2009) trong bối cảnh dữ liệu mất cân bằng. \[3\]  
* **Post-hoc Calibration:** Sử dụng Isotonic Regression để điều chỉnh xác suất dự báo, giải quyết hiện tượng mô hình dự đoán quá tự tin (over-confidence), đặc biệt phổ biến trong các mô hình ensemble. \[4\]
* **Exclusion of SMOTE:** Việc không sử dụng SMOTE là một quyết định có cơ sở lý thuyết rõ ràng, dựa trên phân tích từ Carchiolo & Malgeri (2024). \[2\] Trong bối cảnh không gian đặc trưng thấp, oversampling không tạo thêm thông tin mà chỉ khuếch đại nhiễu, do đó bị loại bỏ khỏi pipeline nhằm tránh “accuracy illusion”.

Cách tiếp cận này nhấn mạnh rằng cải thiện hiệu suất không đến từ việc tăng độ phức tạp mô hình một cách mù quáng, mà từ việc căn chỉnh đúng mục tiêu tối ưu (MCC, calibration) và tôn trọng các giới hạn thông tin của dữ liệu.

## **5\. Evaluation Framework**

Trong bối cảnh dữ liệu mất cân bằng nghiêm trọng và mục tiêu dự báo tập trung vào lớp thiểu số, việc lựa chọn chỉ số đánh giá đóng vai trò quyết định. Thay vì phụ thuộc vào một vài thước đo đơn lẻ, nghiên cứu này xây dựng một framework đa chiều gồm 5 metrics, bao phủ cả khả năng phân biệt (discrimination) và chất lượng xác suất (calibration):
* **Area Under the ROC Curve (AUROC)**: Đo lường khả năng phân biệt tổng thể giữa hai lớp trên toàn bộ không gian ngưỡng. Tuy nhiên, trong dữ liệu mất cân bằng, AUROC có thể đánh giá quá lạc quan.
* **PR-AUC (Area Under the Precision–Recall Curve):** Tập trung trực tiếp vào hiệu suất trên lớp thiểu số. PR-AUC phản ánh rõ ràng hơn sự đánh đổi giữa precision và recall, và nhạy cảm với các cải tiến thực sự trong dự báo tử vong.
* **Matthews Correlation Coefficient (MCC):** Đánh giá mức độ tương quan tổng thể giữa dự đoán và thực tế. Được chọn làm chỉ số trung tâm do khả năng phản ánh cân bằng giữa TP, TN, FP, FN, ngay cả khi dữ liệu bị lệch lớp mạnh.
* **Brier Score:** Đo lường sai số trung bình bình phương giữa xác suất dự báo và kết quả thực tế, cung cấp đánh giá trực tiếp về độ chính xác xác suất. 
* **Expected Calibration Error (ECE):** Đo lường sự sai lệch giữa độ tin cậy của mô hình và độ chính xác thực tế trên các phân đoạn xác suất (bins).

### 5.1. Dummy Baseline as a Reference Point

Để tránh diễn giải sai lệch hiệu suất mô hình, nghiên cứu sử dụng dummy baseline (most-frequent classifier) như một mốc tham chiếu xuyên suốt các cohort. Trên primary cohort, baseline này đạt:

* Accuracy = 92.65%
* PR-AUC = 0.0735 (tương đương prevalence của lớp tử vong)
* Brier Score = 0.0735

Các giá trị này minh họa một reference point quan trọng: một mô hình không có năng lực dự báo vẫn có thể đạt accuracy cao, trong khi các chỉ số như PR-AUC và Brier Score phản ánh đúng bản chất “không thông tin” (non-informative). Do đó, mọi cải thiện của mô hình đều được đánh giá tương đối so với baseline này, thay vì dựa trên giá trị tuyệt đối.

### 5.2. Stratified Evaluation Protocol

Toàn bộ quá trình đánh giá nội bộ được thực hiện với *Stratified K-Fold Cross-Validation*, đảm bảo rằng tỷ lệ lớp (~12.6:1) được bảo toàn trong từng fold. Điều này đặc biệt quan trọng trong bối cảnh dữ liệu mất cân bằng, vì các phương pháp chia ngẫu nhiên thông thường có thể dẫn đến phân phối lớp không ổn định, làm sai lệch kết quả đánh giá.

Ngoài ra, các bước feature engineering nhạy cảm như target encoding cũng được lồng ghép pipeline cross-validation, chỉ học trên training folds và áp dụng cho validation folds, nhằm tránh data leakage và đảm bảo tính hợp lệ của các chỉ số thu được.

## **6\. Results and Analysis**

### **6.1. The Calibration Breakthrough**

Kết quả thực nghiệm cho thấy một hiện tượng nhất quán: hiệu chuẩn xác suất (calibration) không cải thiện đáng kể khả năng phân biệt lớp (AUROC), nhưng tạo ra bước nhảy vọt về chất lượng xác suất dự báo. Điều này được thể hiện rõ ràng trên cả ba cohort:

| Metric | Uncalibrated (Primary CV) | Calibrated (Primary CV) | Calibrated (Study) | Calibrated (Validation/Korea) |
| ------ | ------------------------- | ----------------------- | ------------------ | ----------------------------- |
| Brier  | 0.2246                    | 0.0657                  | 0.1598             | 0.1579                        |
| ECE    | 0.3636                    | 0.0006                  | 0.0934             | 0.1195                        |
| AUROC  | 0.7056                    | 0.7067                  | 0.5894             | 0.5721                        |

Trên primary cohort (cross-validation), Brier Score giảm mạnh từ 0.2246 xuống 0.0657, trong khi ECE gần như bị triệt tiêu (0.3636 → 0.0006), cho thấy xác suất dự báo gần như khớp hoàn hảo với tần suất thực tế. Đáng chú ý, AUROC gần như không thay đổi (+0.0011), xác nhận rằng calibration không làm thay đổi thứ tự xếp hạng (ranking) của mô hình mà chỉ điều chỉnh lại độ tin cậy của xác suất.

Hiệu ứng này tiếp tục duy trì trên các cohort độc lập. Trên study cohort, calibration giúp giảm Brier từ ~0.279 xuống 0.1598 và ECE từ 0.339 xuống 0.0934, cho thấy lợi ích vẫn tồn tại ngay cả khi distribution shift. Trên validation cohort (Hàn Quốc), dù hiệu suất tổng thể suy giảm (AUROC = 0.5721), mô hình vẫn giữ được mức hiệu chuẩn hợp lý (ECE = 0.1195), củng cố vai trò của calibration như một thành phần ổn định xuyên miền.

Những kết quả này cung cấp bằng chứng thực nghiệm rõ ràng cho luận điểm trung tâm của nghiên cứu: trong điều kiện feature bottleneck, việc cải thiện phân tách lớp là rất hạn chế, nhưng chất lượng xác suất dự báo vẫn có thể được cải thiện đáng kể và có ý nghĩa lâm sàng.

### **6.2. Cross-Cohort Performance**

Trên primary cohort, mô hình XGBoost đạt AUROC = 0.7061, gần như tương đương với Logistic Regression đã hiệu chuẩn (0.7067). Sự khác biệt này là không đáng kể về mặt thống kê, cho thấy rằng việc tăng độ phức tạp mô hình không mang lại lợi ích thực chất khi không gian đặc trưng bị giới hạn.

Xu hướng này tiếp tục trên study cohort, nơi Random Forest (AUROC = 0.5945) và XGBoost (0.5931) chỉ nhỉnh hơn rất nhẹ so với Logistic Regression calibrated (0.5894) Mức cải thiện này là quá nhỏ để có ý nghĩa lâm sàng, đặc biệt khi cân nhắc đến chi phí tính toán, khả năng giải thích (interpretability), và độ ổn định của mô hình. Đồng thời, sự sụt giảm hiệu suất trên tập kiểm thử ngoại lai này xác nhận dự đoán của Chicco & Jurman về sự nhạy cảm của các bản ghi lâm sàng tối giản đối với sự thay đổi môi trường y tế. \[1\]

Kết quả này đóng vai trò như một bằng chứng thực nghiệm trực tiếp cho giả thuyết đã được thiết lập trong phần lý thuyết: *Feature Bottleneck* là rào cản mang tính cấu trúc, không thể bị phá vỡ bằng cách tăng độ phức tạp của thuật toán. Trong bối cảnh đó, các mô hình đơn giản nhưng được hiệu chuẩn tốt không chỉ đủ dùng mà còn tối ưu hơn về mặt thực tiễn, đặc biệt trong các ứng dụng y khoa yêu cầu tính minh bạch và độ tin cậy cao.

## **7\. Discussion**

Kết quả nghiên cứu củng cố một chuyển dịch quan trọng trong tư duy xây dựng mô hình y khoa: từ mục tiêu “phân loại đúng” sang mục tiêu “cung cấp xác suất đáng tin cậy”. Trong bối cảnh feature bottleneck, nơi khả năng phân tách lớp bị giới hạn về mặt cấu trúc, việc theo đuổi các cải thiện nhỏ trong AUROC trở nên kém ý nghĩa so với việc đảm bảo rằng xác suất đầu ra phản ánh đúng rủi ro thực tế. Một mô hình có độ hiệu chuẩn cao cho phép bác sĩ lâm sàng sử dụng xác suất đầu ra như một chỉ số rủi ro thực tế thay vì một nhãn nhị phân vô hồn, đặt nền móng cho các hệ thống cảnh báo sớm (Early Warning Systems) có khả năng định lượng sự không chắc chắn (Uncertainty Quantification). \[5\]

Một hướng mở rộng tự nhiên từ framework hiện tại là *Conformal Prediction* (Vovk et al., 2005), đã được nghiên cứu và lên kế hoạch triển khai trong repository. \[5\] Thay vì buộc mô hình đưa ra một quyết định cứng (alive hoặc dead), phương pháp này cho phép xuất ra các tập dự đoán như *{“alive”}*, *{“dead”}*, hoặc *{“alive”, “dead”}* trong trường hợp không chắc chắn. Các trường hợp “uncertain” này có thể được flag để bác sĩ xem xét, tạo ra một cơ chế kết hợp giữa AI và chuyên gia con người một cách có kiểm soát. Cách tiếp cận này không chỉ cải thiện độ an toàn mà còn phù hợp với bản chất không chắc chắn vốn có của dữ liệu lâm sàng.

Tuy nhiên, một giới hạn quan trọng được quan sát là sự suy giảm chất lượng hiệu chuẩn dưới distribution shift. Mặc dù mô hình đạt $ECE = 0.0006$ trên primary cohort, chỉ số này tăng lên $0.0934$ trên study cohort và $0.1195$ trên validation cohort (Hàn Quốc). Điều này cho thấy rằng calibration không hoàn toàn bất biến theo miền dữ liệu, và hiệu chuẩn học được từ một quần thể có thể không chuyển giao hoàn hảo sang quần thể khác. Đây là một bài toán mở (open problem) trong học máy ứng dụng y khoa, gợi ý nhu cầu về các kỹ thuật như domain-adaptive calibration hoặc online recalibration trong môi trường triển khai thực tế.

Một yếu tố quan trọng hỗ trợ cho các hướng phát triển tiếp theo là việc lưu trữ **out-of-fold predictions (OOF)** từ mô hình Logistic Regression dưới dạng file `oof_probabilities_lr.csv`. Tập dữ liệu này cung cấp các xác suất dự báo gần với phân phối thật (unbiased estimates), đóng vai trò như một nền tảng (foundation layer) để xây dựng các phương pháp hậu xử lý nâng cao như conformal prediction hoặc uncertainty quantification mà không cần huấn luyện lại mô hình từ đầu.

Tổng thể, nghiên cứu không chỉ chỉ ra giới hạn của các mô hình hiện tại mà còn mở ra một hướng tiếp cận thực dụng hơn: chấp nhận giới hạn phân tách lớp, nhưng khai thác tối đa giá trị của xác suất và định lượng sự không chắc chắn, từ đó xây dựng các hệ thống hỗ trợ quyết định y khoa an toàn và đáng tin cậy hơn.

## **8\. Conclusion**

Nghiên cứu này đã làm rõ một thực tế quan trọng trong dự báo Sepsis từ dữ liệu lâm sàng tối giản: *giới hạn hiệu suất không nằm ở thuật toán, mà nằm ở thông tin*. Thông qua việc so sánh có hệ thống giữa bốn họ mô hình — Dummy baseline, Logistic Regression (LR), Random Forest (RF), và XGBoost (XGB) — kết quả cho thấy một sự hội tụ nhất quán: không có mô hình nào cải thiện đáng kể khả năng phân biệt lớp (discrimination) vượt qua rào cản của feature bottleneck. Các mô hình phức tạp hơn chỉ mang lại cải thiện biên không đáng kể về AUROC, không đủ để tạo ra giá trị lâm sàng thực chất.

Ngược lại, chúng tôi khẳng định rằng, hiệu chuẩn xác suất (calibration) là bước đi quan trọng nhất để đưa các mô hình AI vào ứng dụng y tế thực tiễn, nơi mà sự tin cậy của xác suất có giá trị cao hơn độ chính xác danh nghĩa. Trong khi discrimination gần như “bão hòa”, calibration cho phép chuyển đổi các mô hình từ trạng thái dự đoán “tự tin sai lệch” sang cung cấp xác suất có ý nghĩa và đáng tin cậy, một yêu cầu cốt lõi trong hỗ trợ quyết định y khoa. Do đó, đóng góp chính của nghiên cứu không phải là một mô hình mới, mà là một tái định nghĩa mục tiêu tối ưu trong bối cảnh dữ liệu bị giới hạn.

Từ góc nhìn ứng dụng, Logistic Regression đã hiệu chuẩn nổi lên như base candidate phù hợp nhất, nhờ sự cân bằng giữa hiệu suất, tính ổn định và khả năng giải thích. Trên cơ sở đó, nghiên cứu đề xuất một lộ trình phát triển rõ ràng: tích hợp thêm Conformal Prediction layer để chuyển từ dự báo xác suất sang dự báo có kiểm soát mức độ không chắc chắn. Cách tiếp cận này hứa hẹn nâng cao độ an toàn của hệ thống bằng cách cho phép mô hình từ chối đưa ra quyết định trong các trường hợp không đủ tự tin.

Quan trọng hơn, hạ tầng cho hướng đi này đã được chuẩn bị: tập tin `oof_probabilities_lr.csv` (out-of-fold predictions) cung cấp các ước lượng xác suất không chệch, đóng vai trò như nền tảng để triển khai các thí nghiệm conformal prediction mà không cần tái huấn luyện mô hình. Điều này mở ra khả năng phát triển nghiên cứu hiện tại thành một công trình tiếp theo chuyên sâu về uncertainty quantification trong y khoa.

Tóm lại, nghiên cứu khẳng định rằng trong các bài toán bị chi phối bởi feature bottleneck, tiến bộ không đến từ việc làm mô hình “phức tạp hơn”, mà từ việc làm cho đầu ra của mô hình trở nên đáng tin cậy hơn. Đây là một chuyển dịch mang tính phương pháp luận, với ý nghĩa trực tiếp đối với việc triển khai AI trong các môi trường lâm sàng thực tế.


## **References**

1. Chicco, D., & Jurman, G. (2020). Survival prediction of patients with sepsis from age, sex, and septic episode number alone. Scientific Reports, 10(1), 17156\.  
2. Carchiolo, V., & Malgeri, M. (2024). Dataset Balancing in Disease Prediction. In Proceedings of the 13th International Conference on Data Science, Technology and Applications \- Volume 1: DATA, 293-300.  
3. He, H., & Garcia, E. A. (2009). Learning from Imbalanced Data. IEEE Transactions on Knowledge and Data Engineering, 21(9), 1263-1284.  
4. Niculescu-Mizil, A., & Caruana, R. (2005). Predicting Good Probabilities with Supervised Learning. In Proceedings of the 22nd International Conference on Machine Learning, 625-632.  
5. Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic Learning in a Random World. Springer.  
6. Cover, T., & Hart, P. (1967). Nearest Neighbor Pattern Classification. IEEE Transactions on Information Theory, 13(1), 21-27.