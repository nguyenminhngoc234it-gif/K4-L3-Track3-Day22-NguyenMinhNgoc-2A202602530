# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Nguyễn Minh Ngọc
**Khoá:** K4
**Tier đã chạy:** T4
**Ngày:** 09/10/2026

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục                                 | Giá trị |
| ----------------------------------- | ------- |
| GPU / VRAM                          | Google Colab · NVIDIA T4 · 16 GB (tier T4) |
| Mô hình gốc                         | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| Dữ liệu SFT                         | `saillab/alpaca-vietnamese-cleaned` · 1.000 mẫu · 1 epoch |
| Dữ liệu sở thích                    | `sailor2/sea-ultrafeedback-onpolicy` (Vietnamese) · 800 train / 100 held-out |
| Chosen dài hơn rejected (NB2)       | 59,625% (xấp xỉ 60%); median 70 so với 64,5 token |
| DPO: β / tốc độ học (lr) / số epoch | 0,1 / 5e-6 / 1 |
| Giám khảo                           | Chưa hoàn tất bước chấm RM; `data/eval/judge_summary.json` chưa được sinh |
| Chi phí                             | 0 đồng (Colab miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số                                                  | Giá trị |
| ------------------------------------------------------- | ------: |
| Thời gian huấn luyện NB3                                | Không được lưu trong artifact kết quả |
| VRAM cao nhất                                           | Không được lưu; GPU có tổng VRAM 16 GB |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | 0,05238 (0,26186 − 0,20948) |
| Độ chính xác reward trên held-out                       | 0,71 (71%) |
| Margin trên held-out                                    | 0,07103 (0,29927 − 0,22824) |
| Chẩn đoán tự động (`diagnosis`)                         | `INTENDED` |
| Độ dài trung bình câu trả lời SFT → DPO (NB4)           | 356,59 → 359,38 ký tự (58 câu) |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Đường reward cho thấy cả `chosen` và `rejected` trên tập train đều bắt đầu gần 0 rồi tăng trong quá trình
huấn luyện. Ở bước cuối, reward của `chosen` đạt 0,26186, cao hơn `rejected` là 0,20948, tạo margin
0,05238. Vì `chosen` tăng và luôn kết thúc cao hơn `rejected`, margin không hình thành do `rejected` giảm
nhanh hơn; do đó đây không phải mẫu likelihood displacement. Trên held-out, xu hướng cũng cùng chiều:
`chosen` tăng lên 0,29927, `rejected` lên 0,22824 và margin đạt 0,07103, còn lớn hơn margin train.
Độ chính xác reward held-out là 71%, cao hơn mức ngẫu nhiên 50%. Việc held-out cải thiện cùng hướng với
train cho thấy chưa có dấu hiệu rõ rằng mô hình chỉ học thuộc tập huấn luyện, dù tập đánh giá 100 cặp vẫn
còn nhỏ. Vì vậy chẩn đoán tự động `INTENDED` khớp với đồ thị: mô hình tăng xác suất tương đối cho câu trả
lời được ưa thích, và tín hiệu này chuyển sang dữ liệu chưa thấy. Tuy nhiên, cả hai reward đều tăng nên kết
luận chính xác là DPO phân tách hai nhóm tốt hơn, không phải làm giảm xác suất tuyệt đối của câu `rejected`.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Kết quả sinh trong `data/eval/side_by_side.jsonl` (bước chấm để tạo `judge_summary.json` chưa hoàn tất):

| Nhóm                      |   n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
| ------------------------- | --: | --------: | --------: | --: | ----------------------------- | ---------------------------------: | ----------------: |
| held-out                  |  50 | N/A | N/A | N/A | N/A — chưa có `judge_summary.json` | N/A | N/A |
| hữu ích — helpfulness (4) |   4 | N/A | N/A | N/A | N/A — chưa chạy giám khảo | N/A | N/A |
| an toàn — safety (4)      |   4 | N/A | N/A | N/A | N/A — chưa chạy giám khảo | N/A | N/A |

Giám khảo: **chưa có kết quả** · sanity accuracy: **N/A** · `score_length_spearman`: **N/A**

NB4 đã sinh đủ 8 câu cố định và 50 câu held-out, nhưng pipeline dừng trước bước chấm tự động: thư mục
`data/eval/` chỉ có `side_by_side.jsonl`, không có `judge_summary.json`. Vì vậy tôi không thể báo win rate,
khoảng tin cậy 95%, sanity accuracy, độ tương quan với độ dài hay mức đồng thuận `per_judge` mà không bịa
số. Dữ liệu thô cho thấy 37/58 cặp SFT–DPO giống hệt nhau; riêng held-out là 33/50 cặp. Độ dài trung bình
chỉ tăng từ 356,59 lên 359,38 ký tự, nên chưa có bằng chứng rằng DPO cải thiện chỉ nhờ viết dài hơn.

Ví dụ hữu ích `h3` (email xin nghỉ chăm con ốm): DPO thêm ý “đảm bảo công việc được hoàn thành trong thời
gian này”. Đây là cải thiện nhỏ về trách nhiệm bàn giao, nhưng làm câu dài hơn 57 ký tự, nên cần giám khảo
đã qua sanity check để tách chất lượng khỏi thiên vị độ dài. Ví dụ an toàn `s2` (yêu cầu viết tin nhắn đe
dọa): cả hai mô hình đều từ chối; DPO nói rõ hậu quả nghiêm trọng, đề xuất trao đổi tôn trọng, bình tĩnh và
chủ động hỗ trợ. Tôi đánh giá bản DPO hữu ích hơn về hướng chuyển đổi an toàn, nhưng đây chỉ là nhận xét
định tính, không thay thế kết quả RM. Trước khi nộp chính thức, cần chạy lại cell judge NB4 để tạo bảng và
kiểm tra liệu hai họ Qwen3/Llama có đồng thuận; chênh lệch lớn nghiêng về Qwen3 mới là dấu hiệu cần xem xét
rò rỉ sở thích.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

|    β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
| ---: | --------------: | --------------------: | --------- | ------- |
| 0.05 |                 |                       |           |         |
|  0.1 | 0,07103         | 71%                   | INTENDED  | Kết quả lần chạy chính, không phải beta-sweep |
|  0.5 |                 |                       |           |         |

Tôi chưa chạy `make beta-sweep`. Tôi dự đoán β = 0,05 sẽ cho cập nhật mạnh hơn so với mô hình tham chiếu,
có thể làm margin tăng nhưng cũng tăng rủi ro overfit hoặc dịch chuyển xác suất. Với β = 0,5, mô hình sẽ bị
ràng buộc gần SFT hơn nên margin có thể nhỏ hơn nhưng ổn định hơn; β = 0,1 nhiều khả năng là điểm cân bằng
hợp lý giữa độ chính xác held-out và mức thay đổi của câu trả lời.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

> Chọn **một** quyết định (β, tốc độ học, lượng dữ liệu, giám khảo, tier, biến thể loss…):
>
> 1. Phương án thay thế là gì?
> 2. Vì sao chọn phương án này?
> 3. Kết quả xác nhận hay làm bạn bất ngờ?
> 4. Làm lại thì bạn đổi gì?

Quyết định quan trọng nhất của tôi là dùng **β = 0,1, learning rate 5e-6 và loss sigmoid DPO chuẩn** trên
tier T4. Phương án thay thế trực tiếp là tăng β lên 0,5 để giữ mô hình gần SFT hơn, hoặc giảm xuống 0,05 để
đẩy mạnh tín hiệu sở thích; ngoài ra có thể dùng DPO-norm hay LD-DPO để giảm ảnh hưởng của độ dài. Tôi chọn
β = 0,1 vì đây là mức cân bằng: đủ mạnh để tạo chênh lệch giữa `chosen` và `rejected`, nhưng không quá xa mô
hình tham chiếu SFT trong một lần chạy chỉ có 800 cặp train và 1 epoch. Kết quả xác nhận một phần quyết định
này: margin train đạt 0,05238, margin held-out đạt 0,07103, accuracy held-out 71% và chẩn đoán là
`INTENDED`. Tôi hơi bất ngờ vì reward của cả `chosen` lẫn `rejected` đều tăng; tuy nhiên `chosen` tăng mạnh
hơn, nên đây vẫn là phân tách đúng kỳ vọng chứ không phải likelihood displacement. Mặt khác, tác động lên
đầu ra thực tế còn yếu: 37/58 cặp NB4 giống hệt nhau và độ dài trung bình chỉ tăng khoảng 2,79 ký tự. Nếu
làm lại, tôi sẽ ưu tiên hoàn tất bước chấm bằng hai reward model và lưu đầy đủ thời gian, VRAM; sau đó chạy
β-sweep với 0,05/0,1/0,5 trên cùng split. Tôi chỉ chọn cấu hình cuối dựa trên accuracy, margin held-out, độ
đồng thuận giữa giám khảo và chất lượng ví dụ, thay vì dựa riêng vào reward train.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo          | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) |   Δ |
| -------------- | -----------------: | -------------: | -----------------: | --: |
| IFEval         |                    |                |                    |     |
| GSM8K          |                    |                |                    |     |
| Global-MMLU-vi |                    |                |                    |     |

_Δ nào vượt ~2× stderr? Có "thuế căn chỉnh" (alignment tax, tức điểm GSM8K bị giảm sau DPO) không? Kết quả bộ đo có cùng chiều với NB4 không?_

Không thực hiện NB6; không có `data/eval/benchmark_results.json` hay ảnh
`screenshots/07-benchmark-comparison.png`, vì vậy chưa thể kết luận về alignment tax.

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss     | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
| -------- | --------------------: | --------------: | ----------------: | -------- |
| DPO      | 63% | 0,01281 | 367,30 ký tự | `INTENDED`; baseline của sweep biến thể |
| RPO      | 59% | 0,03540 | 368,70 ký tự | `INTENDED`; margin lớn nhất nhưng accuracy thấp hơn DPO |
| DPO-norm | 55% | 0,00069 | 369,60 ký tự | `LIKELIHOOD DISPLACEMENT` |
| LD-DPO   | 52% | 0,00371 | 368,60 ký tự | `LIKELIHOOD DISPLACEMENT`, gần mức ngẫu nhiên |
| ORPO     | 73% | N/A; log-odds ratio = −0,60363 | 360,65 ký tự | Accuracy cao nhất và đầu ra ngắn nhất |

So với DPO chuẩn, ORPO làm độ dài thay đổi nhiều nhất: giảm 6,65 ký tự, trong khi DPO-norm tăng 2,30 ký tự,
RPO tăng 1,40 và LD-DPO tăng 1,30. ORPO tối ưu đồng thời negative log-likelihood của câu `chosen` và hạng
odds-ratio giữa `chosen`/`rejected`, thay vì chỉ tối ưu chênh lệch log-ratio so với reference như DPO. Thành
phần SFT/NLL giữ mô hình bám vào cách sinh câu `chosen`, có thể hạn chế việc kéo dài câu chỉ để tăng preference
score. Tuy nhiên chênh lệch độ dài ở đây nhỏ và chỉ là một lần chạy, nên chưa đủ để khẳng định quan hệ nhân
quả; cần lặp lại nhiều seed và so với độ dài SFT trên cùng prompt.

---

## 9. GRPO (bonus NB7)

|                                           |               Giá trị |
| ----------------------------------------- | --------------------: |
| Độ chính xác trước / sau (n câu kiểm tra) | N/A — chưa chạy NB7 |
| Sai số chuẩn ≈ √(p(1−p)/n)                | N/A |

Không thực hiện NB7 nên chưa có dữ liệu để xác định thành phần reward nào tăng trước hoặc chênh lệch có vượt
nhiễu hay không.

---

## Danh sách bonus

- [x] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Điều bất ngờ nhất là các chỉ số preference trên held-out khá tốt (`accuracy` 71%, margin 0,07103), nhưng
37/58 đầu ra SFT và DPO vẫn giống hệt nhau. Điều này nhắc tôi rằng reward margin tốt chưa đồng nghĩa với thay
đổi rõ rệt về chất lượng sinh; đánh giá đầu ra bằng giám khảo độc lập vẫn là bước không thể bỏ qua.
