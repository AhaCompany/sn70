# Custom Miner cho Veridex

Miner này được thiết kế để đạt điểm cao nhất từ validator dựa trên các tiêu chí chấm điểm của Veridex protocol.

## Đặc điểm chính

- **Tìm kiếm đa nguồn**: Kết hợp nhiều API tìm kiếm (Perplexity, Google, Bing) cùng với phương pháp dự phòng
- **Trích xuất thông minh**: Phân tích nội dung trang web để tìm đoạn văn liên quan nhất
- **Đa dạng domain**: Sử dụng nhiều domain khác nhau để tránh bị phạt hệ số domain
- **Phát hiện vô nghĩa**: Nhận diện câu hỏi vô nghĩa để trả về phản hồi trống thay vì bịa chứng cứ
- **Tối ưu hoá tốc độ**: Sử dụng async/await để tìm kiếm và xử lý song song, giảm thời gian phản hồi
- **Lọc chất lượng**: Chọn các đoạn trích rõ ràng hỗ trợ hoặc mâu thuẫn với phát biểu, tránh nội dung trung lập

## Cài đặt

1. Cài đặt dependencies:
```
pip install -r requirements.txt
```

2. Thêm API keys vào file `.env`:
```
PERPLEXITY_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
GOOGLE_CSE_ID=your_cse_id_here
BING_API_KEY=your_key_here
```

## Chạy miner

```
python -m miner.custom.miner
```

## Cấu hình

Bạn có thể điều chỉnh các tham số:

- `--max_snippets`: Số lượng snippet tối đa trả về (mặc định: 5)
- `--max_search_results`: Số kết quả tìm kiếm tối đa (mặc định: 20)
- `--async_timeout`: Thời gian timeout cho xử lý bất đồng bộ (mặc định: 25 giây)