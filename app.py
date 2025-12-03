import streamlit as st
from PIL import Image
from rembg import remove
import io
import zipfile
import numpy as np
import cv2

# --- 頁面設定 ---
st.set_page_config(page_title="莎拉爸貼圖神器 v4.0", page_icon="🐴", layout="wide")
st.title("🐴 莎拉爸貼圖神器 v4.0 (智慧視覺切割版)")
st.markdown("### 終極進化！不再依賴死板網格，使用 AI 視覺技術自動偵測並切割每一張貼圖，保證不切到肉！")

# --- 側邊欄設定 ---
st.sidebar.header("⚙️ 設定參數")
uploaded_file = st.sidebar.file_uploader("請上傳您的貼圖大圖 (JPG/PNG)", type=["jpg", "jpeg", "png"])
st.sidebar.info("💡 v4.0 版本會自動偵測貼圖數量，無需再手動設定行列數。")

# --- 智慧視覺切割演算法 ---
def smart_slice_and_process(image_pil):
    # 1. 將 PIL 圖片轉為 OpenCV 格式 (RGB -> BGR)
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    # 2. 轉為灰階
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # 3. 二值化處理 (黑白分明)，找出物體
    # 使用 Otsu's 方法自動尋找最佳閾值
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 4. 尋找輪廓 (Contours)
    # RETR_EXTERNAL 只找最外層的輪廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 5. 過濾太小的雜訊輪廓
    min_area = 1000 # 可以根據實際情況調整
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    # 6. 根據位置排序 (從上到下，從左到右)
    # 這樣切出來的順序才會是對的
    bounding_boxes = [cv2.boundingRect(c) for c in valid_contours]
    # 先按 y (列) 排序，再按 x (行) 排序，這裡做一個簡單的近似排序
    bounding_boxes.sort(key=lambda x: (round(x[1]/100), x[0]))

    processed_stickers = []
    
    for i, (x, y, w, h) in enumerate(bounding_boxes):
        # 7. 根據輪廓的邊界框切出小圖
        # 為了保險，可以稍微外擴一點點邊界 (padding)
        pad = 5
        x_start, y_start = max(0, x-pad), max(0, y-pad)
        x_end, y_end = min(img_cv.shape[1], x+w+pad), min(img_cv.shape[0], y+h+pad)
        
        sticker_cv = img_cv[y_start:y_end, x_start:x_end]
        
        # 轉回 PIL 格式
        sticker_pil = Image.fromarray(cv2.cvtColor(sticker_cv, cv2.COLOR_BGR2RGB))
        
        # 8. 對切好的小圖進行去背
        sticker_no_bg = remove(sticker_pil)
        
        # 9. 修剪透明空白 (Trim)
        bbox = sticker_no_bg.getbbox()
        if bbox:
            sticker_trimmed = sticker_no_bg.crop(bbox)
            
            # 10. 縮放至 LINE 規格
            target_size = (370, 320)
            sticker_final = sticker_trimmed.copy()
            sticker_final.thumbnail(target_size, Image.Resampling.LANCZOS)
            
            processed_stickers.append(sticker_final)
            
    return processed_stickers

# --- 主邏輯 ---
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="原始大圖預覽", use_container_width=True)
    
    if st.button("🚀 開始智慧切割！"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("⏳ 正在進行智慧視覺分析與切割 (這可能需要一點時間)...")
        progress_bar.progress(20)
        
        try:
            # 執行智慧切割主程序
            stickers = smart_slice_and_process(image)
            total_stickers = len(stickers)
            st.success(f"✅ 成功偵測並切割出 {total_stickers} 張貼圖！")
            progress_bar.progress(50)

            # 準備 ZIP
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                st.write("---")
                st.subheader("👀 最終成品預覽 (前 6 張)")
                preview_cols = st.columns(6)
                
                for i, sticker in enumerate(stickers):
                    count = i + 1
                    current_progress = 50 + (count / total_stickers * 50)
                    progress_bar.progress(int(current_progress))
                    
                    # 存檔
                    img_byte_arr = io.BytesIO()
                    sticker.save(img_byte_arr, format='PNG')
                    zf.writestr(f"{count:02d}.png", img_byte_arr.getvalue())
                    
                    if count <= 6:
                        with preview_cols[count-1]:
                            st.image(sticker, caption=f"{count:02d}.png")
                            
            status_text.text("🎉 所有步驟完成！")
            progress_bar.progress(100)
            
            st.download_button(
                label="📥 下載貼圖包 (ZIP)",
                data=zip_buffer.getvalue(),
                file_name="SarahDad_Stickers_Smart.zip",
                mime="application/zip"
            )
            
        except Exception as e:
            st.error(f"處理過程發生錯誤: {e}")
            st.stop()
