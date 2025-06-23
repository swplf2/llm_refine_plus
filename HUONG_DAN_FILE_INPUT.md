# Hướng Dẫn Sử Dụng File Input - Multi-GPU LLM Refine System

## Tổng Quan

Hệ thống Multi-GPU LLM Refine đã được nâng cấp để hỗ trợ xử lý với file đầu vào, cho phép bạn:

1. **Dịch mới từ file câu nguồn** - Tạo bản dịch hoàn toàn mới
2. **Cải thiện bản dịch có sẵn** - Tinh chỉnh và cải thiện bản dịch đã có

## Các Chế Độ Hoạt Động

### 1. Chế Độ Dịch Mới (Source Only)
Khi chỉ cung cấp file câu nguồn, hệ thống sẽ tạo bản dịch hoàn toàn mới.

```bash
python file_input_refine.py data.en data_translated.vi
```

**Ứng dụng:**
- Dịch tài liệu mới
- Tạo bản dịch chất lượng cao từ đầu
- Xử lý dữ liệu không có bản dịch sẵn

### 2. Chế Độ Cải Thiện (Source + Translation)
Khi cung cấp cả file câu nguồn và bản dịch có sẵn, hệ thống sẽ phân tích và cải thiện bản dịch.

```bash
python file_input_refine.py data.en data_improved.vi --translation-file data_current.vi
```

**Ứng dụng:**
- Cải thiện bản dịch từ Google Translate hoặc hệ thống khác
- Tinh chỉnh bản dịch từ người dịch
- Nâng cao chất lượng bản dịch machine translation

## Định Dạng File

### File Câu Nguồn (VD: data.en)
```
Hello, how are you today?
The weather is beautiful this morning.
I would like to learn Vietnamese language.
Machine translation has improved significantly.
```

**Yêu cầu:**
- Mỗi câu một dòng
- Mã hóa UTF-8
- Định dạng text thuần

### File Bản Dịch (VD: data.vi)
```
Xin chào, hôm nay bạn khỏe không?
Thời tiết đẹp vào buổi sáng này.
Tôi muốn học ngôn ngữ tiếng Việt.
Dịch máy đã cải thiện đáng kể.
```

**Yêu cầu:**
- Mỗi bản dịch một dòng
- Số dòng phải khớp với file nguồn
- Mã hóa UTF-8
- Có thể có dòng trống (sẽ được coi là thiếu bản dịch)

## Ví Dụ Sử Dụng Chi Tiết

### Ví Dụ 1: Dịch Tài Liệu Mới
```bash
# Dịch file tài liệu tiếng Anh sang tiếng Việt
python file_input_refine.py document.en document_vietnamese.vi \
  --num-gpus 4 \
  --max-iterations 5
```

### Ví Dụ 2: Cải Thiện Bản Dịch Google Translate
```bash
# Cải thiện bản dịch từ Google Translate
python file_input_refine.py source.en improved_translation.vi \
  --translation-file google_translate.vi \
  --num-gpus 2 \
  --model llama3.1:8b-instruct-fp16
```

### Ví Dụ 3: Xử Lý Dữ Liệu Lớn
```bash
# Xử lý dataset lớn với nhiều GPU
python file_input_refine.py large_dataset.en refined_dataset.vi \
  --translation-file existing_translations.vi \
  --num-gpus 8 \
  --max-iterations 6 \
  --temperature 50.0 \
  --cooling-rate 0.3
```

## Tùy Chọn Command Line

| Tùy chọn | Viết tắt | Mặc định | Mô tả |
|----------|----------|----------|--------|
| `--translation-file` | `-t` | None | File bản dịch hiện có (tuỳ chọn) |
| `--num-gpus` | `-g` | 4 | Số GPU sử dụng |
| `--model` | `-m` | llama3.1:8b-instruct-fp16 | Model LLM |
| `--max-iterations` | `-i` | 6 | Số lần lặp tối đa |
| `--temperature` | | 41.67 | Nhiệt độ ban đầu |
| `--cooling-rate` | | 0.4 | Tốc độ làm lạnh |

## Quy Trình Xử Lý

### Với Bản Dịch Có Sẵn:
1. **Tải File** - Đọc file nguồn và bản dịch
2. **Phân Tích** - So sánh và đánh giá chất lượng
3. **Cải Thiện** - Tạo bản dịch tốt hơn dựa trên feedback
4. **Tối Ưu** - Sử dụng Simulated Annealing để chọn bản tốt nhất
5. **Lưu Kết Quả** - Xuất bản dịch đã cải thiện

### Không Có Bản Dịch:
1. **Tải File** - Đọc file nguồn
2. **Dịch Mới** - Tạo bản dịch ban đầu
3. **Đánh Giá** - Chấm điểm chất lượng
4. **Tinh Chỉnh** - Cải thiện qua nhiều lần lặp
5. **Lưu Kết Quả** - Xuất bản dịch cuối cùng

## Giám Sát và Hiệu Suất

### Thống Kê Hiệu Suất
Hệ thống cung cấp thông tin chi tiết về:
- Thời gian xử lý tổng
- Số câu được xử lý
- Throughput (câu/giây)
- Hiệu suất trên mỗi GPU
- Tỷ lệ thành công của parsing

### Theo Dõi Tiến Trình
```
🚀 Multi-GPU Generator iteration 1 (Mode: source_and_translation)
🔄 GPU 0 processing 128 sentences in source_and_translation mode...
🔄 GPU 1 processing 128 sentences in source_and_translation mode...
🔄 GPU 2 processing 128 sentences in source_and_translation mode...
🔄 GPU 3 processing 127 sentences in source_and_translation mode...
✅ Multi-GPU generation completed in 45.2s
   Success rate: 98.4%
```

## Xử Lý Lỗi và Khôi Phục

### Lỗi Thường Gặp và Giải Pháp

**1. File không tìm thấy**
```
FileNotFoundError: Source file not found: data.en
```
**Giải pháp:** Kiểm tra đường dẫn file và đảm bảo file tồn tại

**2. Số dòng không khớp**
```
Warning: Source (500) and translation (498) line counts don't match
↳ Padded translation file with empty lines
```
**Giải pháp:** Hệ thống tự động xử lý bằng cách thêm dòng trống

**3. GPU không khả dụng**
```
RuntimeError: No GPUs available
```
**Giải pháp:** Kiểm tra driver NVIDIA và Ollama setup

**4. Lỗi mã hóa**
```
UnicodeDecodeError
```
**Giải pháp:** Hệ thống tự động thử các encoding khác nhau

## Tối Ưu Hiệu Suất

### Cho Dataset Nhỏ (< 1000 câu)
```bash
python file_input_refine.py small.en small_output.vi \
  --num-gpus 1 \
  --max-iterations 3
```

### Cho Dataset Trung Bình (1000-10000 câu)
```bash
python file_input_refine.py medium.en medium_output.vi \
  --num-gpus 2 \
  --max-iterations 4
```

### Cho Dataset Lớn (> 10000 câu)
```bash
python file_input_refine.py large.en large_output.vi \
  --num-gpus 4 \
  --max-iterations 6
```

## Demo và Test

### Chạy Demo
```bash
# Demo đầy đủ với file mẫu
python demo_file_input.py --demo

# Xem yêu cầu hệ thống
python demo_file_input.py --requirements
```

### Tạo File Test
```python
# Tạo file test nhanh
with open("test_source.en", "w", encoding="utf-8") as f:
    f.write("Hello world\n")
    f.write("How are you?\n")
    f.write("Thank you\n")

with open("test_translation.vi", "w", encoding="utf-8") as f:
    f.write("Xin chào thế giới\n")
    f.write("Bạn khỏe không?\n")
    f.write("Cảm ơn bạn\n")
```

## So Sánh Kết Quả

Sau khi xử lý xong, bạn có thể so sánh:
- **File gốc** (nếu có): `input_translation.vi`
- **File đầu ra**: `output_translation.vi`
- **File backup gốc**: `output_translation.vi.original` (khi có bản dịch sẵn)

## Tips và Thủ Thuật

1. **Sử dụng ít GPU hơn** cho dataset nhỏ để tiết kiệm tài nguyên
2. **Tăng số lần lặp** cho chất lượng cao hơn (nhưng chậm hơn)
3. **Kiểm tra file đầu vào** trước khi chạy để tránh lỗi
4. **Backup dữ liệu** quan trọng trước khi xử lý
5. **Monitor GPU usage** để tối ưu hiệu suất

## Troubleshooting

### Hệ thống chậm?
- Giảm số GPU
- Giảm max_iterations
- Kiểm tra RAM và VRAM

### Chất lượng không tốt?
- Tăng max_iterations
- Thử model khác
- Điều chỉnh temperature và cooling_rate

### Lỗi memory?
- Giảm số GPU
- Chia nhỏ file đầu vào
- Tăng virtual memory

## Liên Hệ và Hỗ Trợ

Để được hỗ trợ thêm, vui lòng:
1. Kiểm tra log lỗi chi tiết
2. Đảm bảo đã cài đặt đầy đủ dependencies
3. Kiểm tra documentation và examples
