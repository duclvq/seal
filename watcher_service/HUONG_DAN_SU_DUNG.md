# Hướng dẫn sử dụng hệ thống đóng dấu bản quyền video

---

## Hệ thống làm gì?

Hệ thống **tự động đóng dấu bản quyền vô hình** vào video. Dấu bản quyền được nhúng trực tiếp vào hình ảnh — không nhìn thấy bằng mắt thường, không ảnh hưởng chất lượng video — nhưng có thể truy xuất bất kỳ lúc nào để xác minh nguồn gốc.

---

## Quy trình sử dụng

### Bước 1 — Đưa video vào thư mục INPUT

Copy video cần đóng dấu vào thư mục:

```
\\10.41.185.66\transcode\OUTPUT
```

Hệ thống sẽ **tự động phát hiện** file mới và bắt đầu xử lý. Không cần làm thêm bất kỳ thao tác nào.

> **Lưu ý:** Chờ copy xong hoàn toàn rồi mới đặt vào thư mục. Hệ thống kiểm tra file ổn định trước khi xử lý, tránh bị lỗi giữa chừng.

---

### Bước 2 — Lấy video đã đóng dấu ở thư mục OUTPUT

Sau khi xử lý xong, video đã đóng dấu sẽ xuất hiện tại:

```
\\10.41.185.66\transcode\WATERMARK
```

Tên file và cấu trúc thư mục giữ **nguyên như bản gốc**. File gốc trong INPUT **không bị thay đổi**.

---

### Bước 3 — Theo dõi tiến trình xử lý (Monitor)

Truy cập trang theo dõi tại:

```
http://<địa-chỉ-máy-chủ>:5002
```

Trang này hiển thị trạng thái từng video theo thời gian thực:

| Trạng thái | Ý nghĩa |
|---|---|
| 🟡 **Chờ** (pending) | Video đang chờ đến lượt xử lý |
| 🔵 **Đang xử lý** (processing) | Đang đóng dấu |
| 🟢 **Hoàn thành** (done) | Xong, có thể lấy file ở OUTPUT |
| 🔴 **Lỗi** (error) | Có sự cố — xem cột "Lỗi" để biết nguyên nhân |

**Cột WM Key** hiển thị mã bản quyền đã nhúng vào video đó. Có thể click **copy** để sao chép nhanh.

---

### Bước 4 — Xác minh bản quyền video

Khi cần kiểm tra một video có dấu bản quyền không (ví dụ: video bị rò rỉ), truy cập:

```
http://<địa-chỉ-máy-chủ>:5001
```

Tải video lên, hệ thống sẽ đọc và hiển thị **mã bản quyền** nhúng bên trong.

---

## Câu hỏi thường gặp

**Mất bao lâu để xử lý một video?**
Tùy độ dài và độ phân giải. Thông thường khoảng 1–3 phút cho mỗi video HD 5 phút.

**Video định dạng nào được hỗ trợ?**
MP4, MOV, AVI, MKV, MXF, WebM, FLV, TS, 3GP và các định dạng video phổ biến khác.

**File gốc có bị xóa hoặc thay đổi không?**
Không. File gốc trong thư mục INPUT hoàn toàn không bị chạm đến.

**Nếu đặt cùng một video vào INPUT nhiều lần thì sao?**
Hệ thống tự nhận biết và **bỏ qua**, không xử lý trùng lặp.

**Dấu bản quyền có bị mất nếu nén, cắt ghép video không?**
Dấu được thiết kế để chịu được các tác động thông thường như nén, thay đổi độ sáng, thay đổi tốc độ khung hình. Tuy nhiên, cắt ghép nặng hoặc re-encode nhiều lần liên tiếp có thể làm suy yếu tín hiệu.

**Trang Monitor không mở được?**
Kiểm tra hệ thống đang chạy và đúng địa chỉ IP máy chủ. Liên hệ bộ phận kỹ thuật nếu vẫn không truy cập được.

---

## Liên hệ hỗ trợ

Nếu gặp sự cố, cung cấp cho bộ phận kỹ thuật:
- Tên file video bị lỗi
- Ảnh chụp màn hình trang Monitor (cột Lỗi)
- Thời điểm xảy ra sự cố
