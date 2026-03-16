import os
import sys
from predict import VietNameOCR

def main():
    # 1. Khai báo đường dẫn đến cấu hình và trọng số
    # (Đường dẫn tương đối so với vị trí file THUC_NGHIEM.py)
    config_path = 'config_mobilenet_svtr_ctc.yml'
    weights_path = 'weights/mobilenet_svtr_ctc.pth'
    
    # Đường dẫn ảnh test (bạn có thể thay bằng đường dẫn ảnh thực tế của bạn)
    image_path = '1.jpg'
    
    # 2. Nạp mô hình (Nên để ở ngoài các vòng lặp hoặc hàm route của API để tránh nạp lại nhiều lần)
    print("Đang nạp mô hình VietNameOCR...")
    try:
        ocr_engine = VietNameOCR(config_path, weights_path)
        print("✅ Mô hình đã được nạp thành công!\n")
    except Exception as e:
        print(f"❌ Lỗi khi nạp mô hình: {e}")
        sys.exit(1)
        
    # 3. Tiến hành nhận diện
    print(f"Đang nhận diện ảnh: {image_path}")
    
    # Kiểm tra xem ảnh có tồn tại không (nếu không tạo 1 ảnh trắng tạm để test tool)
    if not os.path.exists(image_path):
        print(f"⚠️ Cảnh báo: Không tìm thấy file '{image_path}'!")
        print("   -> Tự động tạo một ảnh trống tên 'test_image.jpg' để chạy thử luồng code...")
        from PIL import Image
        img = Image.new('RGB', (200, 50), color='white')
        img.save(image_path)
        
    try:
        # Gọi hàm predict truyền vào đường dẫn ảnh
        ket_qua = ocr_engine.predict(image_path)
        
        print("-" * 40)
        print(f"KẾT QUẢ OCR: {ket_qua}")
        print("-" * 40)
    except Exception as e:
        print(f"❌ Lỗi trong quá trình dự đoán: {e}")

if __name__ == "__main__":
    main()
