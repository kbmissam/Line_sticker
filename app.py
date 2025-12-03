import streamlit as st
from PIL import Image
from rembg import remove
import io
import zipfile
import numpy as np

# --- 頁面設定 ---
st.set_page_config(page_title="莎拉爸貼圖神器 v3.0", page_icon="🐴", layout="wide")
st.title("🐴 莎拉爸貼圖神器 v3.0 (綠幕終極版)")
st.markdown("### 新增功能：針對「螢光綠」背景的專用切除模式，不再依賴 AI 猜測！")

# --- 側邊欄設定 ---
st.sidebar.header("⚙️ 設定參數")
uploaded_file = st.sidebar.file_uploader("請上傳您的貼圖大圖 (JPG/PNG)", type=["jpg", "jpeg", "png"])
rows = st.sidebar.number_input("縱向 (Rows)", min_value=1, value=5)
cols = st.sidebar.number_input("橫向 (Columns)", min_value=1, value=6)

# --- ⭐ 新增模式切換 ⭐ ---
st.sidebar.markdown("---")
st.sidebar.header("🎨 去背模式選擇")
remove_mode = st.sidebar.radio(
    "請選擇去背方式：",
    ("🟢 綠幕模式 (Chroma Key) - 推薦綠底圖用", "🤖 AI 模式 (Rembg) - 一般白底圖用")
)

# --- 綠幕去背演算法 (不靠 AI，靠數學) ---
def remove_green_screen_math(img_pil):
    # 轉成陣列
    img = np.array(img_pil.convert("RGBA"))
    # 分離通道
    r, g, b, a = img[:, :, 0], img[:, :, 1], img[:, :, 2], img[:, :, 3]
    
    # 定義「綠色」：綠色通道數值很高，且明顯大於紅藍通道
    # 這裡的數值可以微調，但對螢光綠通常很準
    # 條件：Green > 100 且 Green > Red + 20 且 Green > Blue + 20
    mask = (g > 100) & (g > r + 30) & (g > b + 30)
    
    # 將符合條件(綠色)的像素，Alpha 設為 0 (透明)
    img[mask, 3] = 0
    
    return Image.fromarray(img)

# --- 主邏輯 ---
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGBA")
    st.image(image, caption="原始大圖預覽", use_container_width=True)
    
    if st.button("🚀 開始魔法處理！"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # --- 步驟 1: 先去背 (根據選擇的模式) ---
        status_text.text("⏳ 正在進行去背處理...")
        progress_bar.progress(10)
        
        try:
            if "綠幕模式" in remove_mode:
                # 使用物理數學法
                image_no_bg = remove_green_screen_math(image)
                st.success("✅ 已使用綠幕物理切除法")
            else:
                # 使用原本的 AI 法
                image_no_bg = remove(image)
                st.success("✅ 已使用 AI 智慧去背")
                
            st.image(image_no_bg, caption="去背後預覽 (檢查這裡！)", use_container_width=True)
            
        except Exception as e:
            st.error(f"去背過程發生錯誤: {e}")
            st.stop()

        status_text.text("✅ 去背完成！開始切割...")
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
                    
                    # 裁切
                    left = c * cell_width
                    upper = r * cell_height
                    right = left + cell_width
                    lower = upper + cell_height
                    sticker = image_no_bg.crop((left, upper, right, lower))
                    
                    # 修剪透明空白 (Trim)
                    bbox = sticker.getbbox()
                    if bbox:
                        sticker_trimmed = sticker.crop(bbox)
                        
                        # 縮放
                        target_size = (370, 320)
                        sticker_final = sticker_trimmed.copy()
                        sticker_final.thumbnail(target_size, Image.Resampling.LANCZOS)
                        
                        # 存檔
                        img_byte_arr = io.BytesIO()
                        sticker_final.save(img_byte_arr, format='PNG')
                        zf.writestr(f"{count:02d}.png", img_byte_arr.getvalue())
                        
                        if count <= 6:
                            with preview_cols[count-1]:
                                st.image(sticker_final, caption=f"{count:02d}.png")
                    else:
                        pass 

        status_text.text("🎉 處理完成！")
        progress_bar.progress(100)
        
        st.download_button(
            label="📥 下載貼圖包 (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="SarahDad_Stickers_Green.zip",
            mime="application/zip"
        )
