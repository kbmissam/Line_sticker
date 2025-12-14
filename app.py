import streamlit as st
from PIL import Image
from rembg import remove
import io
import zipfile
import numpy as np
import cv2

# --- 頁面設定 (版本號更新) ---
st.set_page_config(page_title="莎拉爸貼圖神器 v6.0", page_icon="🐴", layout="wide")
st.title("🐴 莎拉爸貼圖神器 v6.0 (自由網格版)")

# --- 側邊欄：功能導航 ---
st.sidebar.header("🚀 功能選擇")
app_mode = st.sidebar.radio(
    "請問您想做什麼？",
    ("✂️ 貼圖自動切片", "🖼️ 製作主要與標籤圖片")
)

st.sidebar.markdown("---")

# ==========================================
# 功能 A：貼圖自動切片
# ==========================================
if app_mode == "✂️ 貼圖自動切片":
    st.markdown("### 步驟 1：上傳貼圖大圖 (推薦綠底)。")
    
    # --- 側邊欄設定 ---
    st.sidebar.header("⚙️ 參數設定")
    uploaded_file = st.sidebar.file_uploader("請上傳貼圖大圖 (JPG/PNG)", type=["jpg", "jpeg", "png"])

    st.sidebar.header("🎨 去背模式")
    remove_mode = st.sidebar.radio(
        "選擇去背方式：",
        ("🟢 綠幕模式 (推薦！)", "🤖 AI 模式 (白底用)")
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("📐 切割策略 (關鍵！)")
    # v6.0 核心更新：讓使用者選擇切割邏輯
    slice_mode = st.sidebar.radio(
        "選擇切割方式：",
        ("🧠 智慧視覺偵測 (預設)", "📏 強制網格切割 (自訂行烈)")
    )

    if slice_mode == "🧠 智慧視覺偵測 (預設)":
        dilation_size = st.sidebar.slider("膨脹係數 (防切字)", 5, 50, 25)
        st.sidebar.info("💡 適合：排列不規則，或背景很乾淨的圖。")
    else:
        # v6.0 新增：自由設定行列數
        st.sidebar.warning("⚠️ 強制模式：請設定原圖的網格數量。")
        col_r, col_c = st.sidebar.columns(2)
        with col_r:
            rows = st.number_input("縱向列數 (Rows)", min_value=1, value=5, step=1)
        with col_c:
            cols = st.number_input("橫向行數 (Columns)", min_value=1, value=6, step=1)
        st.sidebar.success(f"目前設定：將切割為 {rows} x {cols} = {rows*cols} 張")

    # --- 綠幕演算法 ---
    def remove_green_screen_math(img_pil):
        img = np.array(img_pil.convert("RGBA"))
        r, g, b, a = img[:, :, 0], img[:, :, 1], img[:, :, 2], img[:, :, 3]
        # 稍微放寬一點綠色容許度，避免水彩邊緣沒去乾淨
        mask = (g > 90) & (g > r + 15) & (g > b + 15)
        img[mask, 3] = 0
        return Image.fromarray(img)

    # --- 核心處理邏輯 ---
    def process_image(image_pil, mode_selection, slicing_strategy, dilation_val=25, grid_rows=5, grid_cols=6):
        img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        processed_stickers = []
        
        # 策略 A: 智慧視覺偵測
        if "智慧" in slicing_strategy:
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

            for x, y, w, h in bounding_boxes:
                # 切割
                sticker_cv = img_cv[y:y+h, x:x+w]
                sticker_pil = Image.fromarray(cv2.cvtColor(sticker_cv, cv2.COLOR_BGR2RGB))
                
                # 去背
                if "綠幕" in mode_selection:
                    sticker_no_bg = remove_green_screen_math(sticker_pil)
                else:
                    sticker_no_bg = remove(sticker_pil)
                
                # Trim & Resize
                bbox = sticker_no_bg.getbbox()
                if bbox:
                    sticker_trimmed = sticker_no_bg.crop(bbox)
                    sticker_final = sticker_trimmed.copy()
                    sticker_final.thumbnail((370, 320), Image.Resampling.LANCZOS)
                    processed_stickers.append(sticker_final)

        # 策略 B: 強制網格切割 (v6.0 更新使用變數)
        else:
            height, width, _ = img_cv.shape
            cell_h = height // grid_rows # 使用設定的列數
            cell_w = width // grid_cols # 使用設定的行數
            
            for r in range(grid_rows):
                for c in range(grid_cols):
                    x = c * cell_w
                    y = r * cell_h
                    
                    # 簡單粗暴：切！
                    sticker_cv = img_cv[y:y+cell_h, x:x+cell_w]
                    sticker_pil = Image.fromarray(cv2.cvtColor(sticker_cv, cv2.COLOR_BGR2RGB))
                    
                    # 切完後再單獨去背
                    if "綠幕" in mode_selection:
                        sticker_no_bg = remove_green_screen_math(sticker_pil)
                    else:
                        sticker_no_bg = remove(sticker_pil)
                    
                    # Trim & Resize
                    bbox = sticker_no_bg.getbbox()
                    if bbox:
                        sticker_trimmed = sticker_no_bg.crop(bbox)
                        sticker_final = sticker_trimmed.copy()
                        sticker_final.thumbnail((370, 320), Image.Resampling.LANCZOS)
                        processed_stickers.append(sticker_final)

        return processed_stickers

    # --- 主邏輯 UI ---
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=f"原始大圖 ({image.width}x{image.height})", use_container_width=True)
        
        if st.button("🚀 開始處理！"):
            progress_bar = st.progress(0)
            
            try:
                # 判斷傳入參數
                d_val = dilation_size if "智慧" in slice_mode else 0
                # v6.0 傳入行列數
                r_val = rows if "強制" in slice_mode else 5
                c_val = cols if "強制" in slice_mode else 6

                stickers = process_image(image, remove_mode, slice_mode, d_val, r_val, c_val)
                total_stickers = len(stickers)
                
                if total_stickers == 0:
                    st.error("⚠️ 處理失敗，找不到貼圖！")
                    st.stop()
                    
                st.success(f"✅ 成功切割出 {total_stickers} 張貼圖！")
                progress_bar.progress(100)

                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    st.subheader(f"👀 預覽 (前 {min(grid_cols, total_stickers)} 張)")
                    preview_cols_ui = st.columns(min(grid_cols, total_stickers)) # 依據行數調整預覽
                    for i, sticker in enumerate(stickers):
                        count = i + 1
                        img_byte_arr = io.BytesIO()
                        sticker.save(img_byte_arr, format='PNG')
                        zf.writestr(f"{count:02d}.png", img_byte_arr.getvalue())
                        if i < len(preview_cols_ui):
                            with preview_cols_ui[i]:
                                st.image(sticker, caption=f"{count:02d}.png")
                
                st.download_button(
                    label="📥 下載貼圖包 (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name=f"SarahDad_Stickers_{total_stickers}pcs.zip",
                    mime="application/zip"
                )
            except Exception as e:
                st.error(f"錯誤: {e}")

# ==========================================
# 功能 B：製作主要與標籤圖片
# ==========================================
elif app_mode == "🖼️ 製作主要與標籤圖片":
    st.markdown("### 步驟 2：製作上架專用縮圖。")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1️⃣ Main (240x240)")
        main_file = st.file_uploader("上傳一張 PNG", type=["png"], key="main")
        if main_file:
            img = Image.open(main_file).convert("RGBA")
            img.thumbnail((240, 240), Image.Resampling.LANCZOS)
            bg = Image.new("RGBA", (240, 240), (0,0,0,0))
            bg.paste(img, ((240-img.width)//2, (240-img.height)//2))
            st.image(bg)
            buf = io.BytesIO()
            bg.save(buf, format="PNG")
            st.download_button("下載 Main", buf.getvalue(), "main.png", "image/png")
            
    with col2:
        st.subheader("2️⃣ Tab (96x74)")
        tab_file = st.file_uploader("上傳一張 PNG", type=["png"], key="tab")
        if tab_file:
            img = Image.open(tab_file).convert("RGBA")
            img.thumbnail((96, 74), Image.Resampling.LANCZOS)
            bg = Image.new("RGBA", (96, 74), (0,0,0,0))
            bg.paste(img, ((96-img.width)//2, (74-img.height)//2))
            st.image(bg)
            buf = io.BytesIO()
            bg.save(buf, format="PNG")
            st.download_button("下載 Tab", buf.getvalue(), "tab.png", "image/png")
