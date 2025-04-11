# So sánh Miners

## Hướng dẫn cài đặt và chạy

1. Cài đặt thư viện:
```
pip install -r requirements.txt
```

2. Thêm API key vào file `.env`:
```
OPENAI_API_KEY=your_openai_key_here
PERPLEXITY_API_KEY=your_perplexity_key_here
```

3. Chạy các miners:

```bash
# Chạy OpenAI miner
python -m miner.openai_miner.miner

# Chạy Perplexity miner
python -m miner.perplexity_miner.miner
```

## So sánh hiệu quả

Để so sánh hiệu quả của hai miners, hãy theo dõi:

1. **Tỷ lệ thành công**: Tỷ lệ trả về bằng chứng hợp lệ cho các truy vấn
2. **Điểm validator**: Các điểm số được validator gán cho mỗi miner
3. **Tốc độ phản hồi**: Thời gian phản hồi trung bình 
4. **Chất lượng nội dung**: Đánh giá định tính về mức độ liên quan của bằng chứng
5. **Chi phí API**: Phí API cho mỗi miner

Để theo dõi điểm được validator gán, quan sát giá trị `Incentive` trong logs của miner hoặc sử dụng lệnh sau để xem giá trị incentive trên blockchain:

```bash
btcli s list --netuid 1 --subtensor.network finney
```

## Cải tiến

Sau khi xác định miner có hiệu suất tốt hơn, bạn có thể cải tiến nó bằng cách:

1. Tinh chỉnh các prompts
2. Điều chỉnh cơ chế phát hiện nội dung vô nghĩa
3. Tối ưu hóa logic lọc và đa dạng hóa domain
4. Điều chỉnh tham số nhiệt độ (temperature) cho những câu trả lời tốt hơn
5. Thử nghiệm các mô hình khác nhau (gpt-4o, gpt-4-turbo, sonar-small-chat, sonar-pro, v.v.)

Ghi chú đánh giá hiệu suất của bạn trong file README này để theo dõi tiến độ qua thời gian.