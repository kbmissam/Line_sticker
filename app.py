import streamlit as st
from PIL import Image
from rembg import remove
import io
import zipfile
import numpy as np
import cv2

# --- 頁面設定 ---
st.set_page_config(page_title="莎拉爸貼圖神器 v4.2", page_icon="🐴", layout="wide")
st.title("🐴 莎拉爸貼圖神器 v4.2 (智慧切割+膨脹修復版)")
st.markdown("### v4.2 更新：加入「視覺膨脹技術」，確保文字框與角色緊密相連，不再切到肉！")

# --- 側邊欄設定 ---
st.sidebar.header("⚙️ 設定參數")
uploaded_file = st.sidebar.file_uploader("請上傳您的貼圖大圖 (JPG/PNG)", type=["jpg", "jpeg", "png"])

st.sidebar.markdown("---")
st.sidebar.header("🎨 去背模式選擇")
remove_mode = st.sidebar.radio(
    "請選擇去背方式：",
    ("🟢 綠幕模式 (推薦！物理切除)", "🤖 AI 模式 (一般白底圖用)")
)

# 新增參數：讓使用者可以手動調整「膨脹程度」
st.sidebar.markdown("---")
st.sidebar.header("🔧 進階微調")
dilation_size = st.sidebar.slider("膨脹係數 (切到字請調大，黏在一起請調小)", 5, 50, 25)

st.sidebar.info("💡 系統會自動偵測貼圖數量與位置，無需手動設定行列。")

# --- 核心演算法 1：綠幕物理去背 ---
def remove_green_screen_math(img_pil):
    img = np.array(img_pil.convert("RGBA"))
    r, g, b, a = img[:, :, 0], img[:, :, 1], img[:, :, 2], img[:, :, 3]
    mask = (g > 100) & (g > r + 20) & (g > b + 20)
    img[mask, 3] = 0
    return Image.fromarray(img)

# --- 核心演算法 2：智慧視覺切割主程序 ---
def smart_process(image_pil, mode_selection, dilation_val):
    # 1. 準備 OpenCV 格式
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # 2. 二值化 (黑白分明)
    if "綠幕" in mode_selection:
        # 綠幕通常比較暗(轉灰階後)，Otsu 能抓得很好
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        # 白底圖反轉
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # === ⭐ v4.2 關鍵修復：膨脹 (Dilation) ⭐ ===
    # 這裡讓物體「變胖」，把文字跟馬黏在一起，確保視為同一個物體
    # kernel 大小由側邊欄滑桿控制 (預設 25x25)
    kernel = np.ones((dilation_val, dilation_val), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    # ==========================================

    # 3. 尋找輪廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 4. 過濾雜訊
    min_area = 1000 
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    # 5. 排序
    bounding_boxes = [cv2.boundingRect(c) for c in valid_contours]
    bounding_boxes.sort(key=lambda x: (round(x[1]/100), x[0]))

    processed_stickers = []
    
    for i, (x, y, w, h) in enumerate(bounding_boxes):
        # 6. 切割 (基於膨脹後的框，再稍微縮一點點回來或保持原樣)
        # 因為膨脹會讓框變很大，這裡我們不需要額外加太多的 padding 了
        pad = 0 
        x_start, y_start = max(0, x-pad), max(0, y-pad)
        x_end, y_end = min(img_cv.shape[1], x+w+pad), min(img_cv.shape[0], y+h+pad)
        
        sticker_cv = img_cv[y_start:y_end, x_start:x_end]
        sticker_pil = Image.fromarray(cv2.cvtColor(sticker_cv, cv2.COLOR_BGR2RGB))
        
        # 7. 去背
        if "綠幕" in mode_selection:
            sticker_no_bg = remove_green_screen_math(sticker_pil)
        else:
            sticker_no_bg = remove(sticker_pil)
        
        # 8. Trim
        bbox = sticker_no_bg.getbbox()
        if bbox:
            sticker_trimmed = sticker_no_bg.crop(bbox)
            
            # 9. Resize
            target_size = (370, 320)
            sticker_final = sticker_trimmed.copy()
            sticker_final.thumbnail(target_size, Image.Resampling.LANCZOS)
            
            processed_stickers.append(sticker_final)
            
    return processed_stickers

# --- 主程式邏輯 ---
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="原始大圖預覽", use_container_width=True)
    
    if st.button("🚀 開始智慧處理！"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text(f"⏳ 正在進行智慧膨脹偵測...")
        progress_bar.progress(10)
        
        try:
            # 傳入膨脹係數
            stickers = smart_process(image, remove_mode, dilation_size)
            
            total_stickers = len(stickers)
            if total_stickers == 0:
                st.error("⚠️ 找不到貼圖！請嘗試調整膨脹係數。")
                st.stop()
                
            st.success(f"✅ 成功切割出 {total_stickers} 張貼圖！(如果不準，請調整左側膨脹係數)")
            progress_bar.progress(50)

            # 打包
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                st.write("---")
                st.subheader("👀 最終成品預覽 (前 6 張)")
                preview_cols = st.columns(6)
                
                for i, sticker in enumerate(stickers):
                    count = i + 1
                    current_progress = 50 + (count / total_stickers * 50)
                    progress_bar.progress(int(current_progress))
                    
                    img_byte_arr = io.BytesIO()
                    sticker.save(img_byte_arr, format='PNG')
                    zf.writestr(f"{count:02d}.png", img_byte_arr.getvalue())
                    
                    if count <= 6:
                        with preview_cols[count-1]:
                            st.image(sticker, caption=f"{count:02d}.png")
                            
            status_text.text("🎉 完美處理完成！")
            progress_bar.progress(100)
            
            st.download_button(
                label="📥 下載貼圖包 (ZIP)",
                data=zip_buffer.getvalue(),
                file_name="SarahDad_Stickers_v4.2.zip",
                mime="application/zip"
            )
            
        except Exception as e:
            st.error(f"處理過程發生錯誤: {e}")
