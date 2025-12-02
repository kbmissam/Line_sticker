import streamlit as st
from PIL import Image
from rembg import remove
import io
import zipfile
import numpy as np

# --- 頁面設定 ---
st.set_page_config(page_title="莎拉爸貼圖神器 v2.0", page_icon="🐴", layout="wide")
st.title("🐴 莎拉爸切圖神器 v2.0 (優化去背版)")
st.markdown("### 核心邏輯升級：先整張大圖智慧去背，再進行精準切割。")

# --- 側邊欄設定 ---
st.sidebar.header("⚙️ 設定參數")
uploaded_file = st.sidebar.file_uploader("請上傳您的貼圖大圖 (JPG/PNG)", type=["jpg", "jpeg", "png"])
rows = st.sidebar.number_input("縱向 (Rows) - 直的有幾排?", min_value=1, value=5)
cols = st.sidebar.number_input("橫向 (Columns) - 橫的有幾個?", min_value=1, value=6)

# --- 主邏輯 ---
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGBA") # 確保是 RGBA 模式
    st.image(image, caption="原始大圖預覽", use_container_width=True)
    
    if st.button("🚀 開始魔法處理 (v2.0)！"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # --- 步驟 1: 整張大圖先去背 (關鍵改進) ---
        status_text.text("⏳ 正在進行整張大圖 AI 去背 (這可能需要一點時間)...")
        progress_bar.progress(10)
        
        # 這裡可以加入參數調整去背強度，目前先用預設
        try:
            image_no_bg = remove(image)
            st.image(image_no_bg, caption="整張去背預覽 (檢查這裡有沒有破圖)", use_container_width=True)
        except Exception as e:
            st.error(f"去背過程發生錯誤: {e}")
            st.stop()

        status_text.text("✅ 大圖去背完成！開始切割...")
        progress_bar.progress(30)

        # 準備 ZIP
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            width, height = image_no_bg.size
            cell_width = width / cols
            cell_height = height / rows
            
            total_stickers = rows * cols
            count = 0
            
            st.write("---")
            st.subheader("👀 最終成品預覽 (前 6 張)")
            preview_cols = st.columns(6)
            
            for r in range(rows):
                for c in range(cols):
                    count += 1
                    current_progress = 30 + (count / total_stickers * 70)
                    progress_bar.progress(int(current_progress))
                    
                    # --- 步驟 2: 裁切已去背的大圖 ---
                    left = c * cell_width
                    upper = r * cell_height
                    right = left + cell_width
                    lower = upper + cell_height
                    
                    # 這裡很重要：要切「去背後」的那張圖
                    sticker = image_no_bg.crop((left, upper, right, lower))
                    
                    # --- 步驟 3: 修剪透明空白 (Trim) ---
                    # 檢查 Alpha 通道是否全透明
                    if sticker.getbbox():
                        sticker_trimmed = sticker.crop(sticker.getbbox())
                        
                        # --- 步驟 4: 縮放至 LINE 規格 ---
                        target_size = (370, 320)
                        sticker_final = sticker_trimmed.copy()
                        sticker_final.thumbnail(target_size, Image.Resampling.LANCZOS)
                        
                        # --- 步驟 5: 存入 ZIP ---
                        img_byte_arr = io.BytesIO()
                        sticker_final.save(img_byte_arr, format='PNG')
                        zf.writestr(f"{count:02d}.png", img_byte_arr.getvalue())
                        
                        if count <= 6:
                            with preview_cols[count-1]:
                                st.image(sticker_final, caption=f"{count:02d}.png")
                    else:
                        pass # 空圖跳過

        status_text.text("🎉 所有步驟完成！")
        progress_bar.progress(100)
        st.success(f"成功處理！請下載 ZIP 檔。")
        
        st.download_button(
            label="📥 下載 v2.0 貼圖包 (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="SarahDad_Stickers_v2.zip",
            mime="application/zip"
        )
