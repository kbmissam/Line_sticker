import streamlit as st
from PIL import Image
from rembg import remove
import io
import zipfile
import numpy as np
import cv2

# --- 頁面設定 ---
st.set_page_config(page_title="莎拉爸貼圖神器 v5.0", page_icon="🐴", layout="wide")
st.title("🐴 莎拉爸貼圖神器 v5.0 (全能上架版)")

# --- 側邊欄：功能導航 ---
st.sidebar.header("🚀 功能選擇")
app_mode = st.sidebar.radio(
    "請問您想做什麼？",
    ("✂️ 貼圖自動切片", "🖼️ 製作主要與標籤圖片")
)

st.sidebar.markdown("---")

# ==========================================
# 功能 A：貼圖自動切片 (原本的核心功能)
# ==========================================
if app_mode == "✂️ 貼圖自動切片":
    st.markdown("### 步驟 1：上傳 Midjourney 生成的綠底大圖，自動切成 30-40 張小貼圖。")
    
    # --- 側邊欄設定 ---
    st.sidebar.header("⚙️ 切片參數設定")
    uploaded_file = st.sidebar.file_uploader("請上傳貼圖大圖 (JPG/PNG)", type=["jpg", "jpeg", "png"])

    st.sidebar.header("🎨 去背模式")
    remove_mode = st.sidebar.radio(
        "選擇去背方式：",
        ("🟢 綠幕模式 (推薦！)", "🤖 AI 模式 (白底用)")
    )
    
    st.sidebar.header("🔧 進階微調")
    dilation_size = st.sidebar.slider("膨脹係數 (防切字)", 5, 50, 25)
    
    st.sidebar.info("💡 系統會自動偵測貼圖數量，無需設定行列。")

    # --- 綠幕演算法 ---
    def remove_green_screen_math(img_pil):
        img = np.array(img_pil.convert("RGBA"))
        r, g, b, a = img[:, :, 0], img[:, :, 1], img[:, :, 2], img[:, :, 3]
        mask = (g > 100) & (g > r + 20) & (g > b + 20)
        img[mask, 3] = 0
        return Image.fromarray(img)

    # --- 智慧切割主程序 ---
    def smart_process(image_pil, mode_selection, dilation_val):
        img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        if "綠幕" in mode_selection:
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((dilation_val, dilation_val), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 1000 
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        bounding_boxes = [cv2.boundingRect(c) for c in valid_contours]
        bounding_boxes.sort(key=lambda x: (round(x[1]/100), x[0]))

        processed_stickers = []
        for i, (x, y, w, h) in enumerate(bounding_boxes):
            pad = 0 
            x_start, y_start = max(0, x-pad), max(0, y-pad)
            x_end, y_end = min(img_cv.shape[1], x+w+pad), min(img_cv.shape[0], y+h+pad)
            
            sticker_cv = img_cv[y_start:y_end, x_start:x_end]
            sticker_pil = Image.fromarray(cv2.cvtColor(sticker_cv, cv2.COLOR_BGR2RGB))
            
            if "綠幕" in mode_selection:
                sticker_no_bg = remove_green_screen_math(sticker_pil)
            else:
                sticker_no_bg = remove(sticker_pil)
            
            bbox = sticker_no_bg.getbbox()
            if bbox:
                sticker_trimmed = sticker_no_bg.crop(bbox)
                target_size = (370, 320)
                sticker_final = sticker_trimmed.copy()
                sticker_final.thumbnail(target_size, Image.Resampling.LANCZOS)
                processed_stickers.append(sticker_final)
        return processed_stickers

    # --- 主邏輯 (切片) ---
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="原始大圖", use_container_width=True)
        
        if st.button("🚀 開始智慧切片！"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("⏳ 處理中...")
            
            try:
                stickers = smart_process(image, remove_mode, dilation_size)
                total_stickers = len(stickers)
                
                if total_stickers == 0:
                    st.error("⚠️ 找不到貼圖！請調整膨脹係數。")
                    st.stop()
                    
                st.success(f"✅ 成功切割出 {total_stickers} 張貼圖！")
                progress_bar.progress(100)

                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    st.subheader("👀 預覽")
                    preview_cols = st.columns(6)
                    for i, sticker in enumerate(stickers):
                        count = i + 1
                        img_byte_arr = io.BytesIO()
                        sticker.save(img_byte_arr, format='PNG')
                        zf.writestr(f"{count:02d}.png", img_byte_arr.getvalue())
                        if count <= 6:
                            with preview_cols[count-1]:
                                st.image(sticker, caption=f"{count:02d}.png")
                
                st.download_button(
                    label="📥 下載貼圖包 (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name="SarahDad_Stickers_v5.zip",
                    mime="application/zip"
                )
            except Exception as e:
                st.error(f"錯誤: {e}")

# ==========================================
# 功能 B：製作主要與標籤圖片 (新功能)
# ==========================================
elif app_mode == "🖼️ 製作主要與標籤圖片":
    st.markdown("### 步驟 2
