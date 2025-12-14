import streamlit as st
from PIL import Image
from rembg import remove
import io
import zipfile
import numpy as np
import cv2

# --- 頁面設定 ---
st.set_page_config(page_title="莎拉爸貼圖神器 v6.1", page_icon="🐴", layout="wide")
st.title("🐴 莎拉爸貼圖神器 v6.1 (批量流水線版)")
st.markdown("支援 **多圖同時上傳**，並將結果**自動合併編號**打包成一個 ZIP！")

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
    st.markdown("### 步驟 1：上傳貼圖大圖 (可多選)。")
    
    # --- 側邊欄設定 ---
    st.sidebar.header("⚙️ 參數設定")
    # v6.1 更新：accept_multiple_files=True
    uploaded_files = st.sidebar.file_uploader(
        "請上傳貼圖大圖 (JPG/PNG)，可按住 Ctrl 或 Shift 多選", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )

    st.sidebar.header("🎨 去背模式")
    remove_mode = st.sidebar.radio(
        "選擇去背方式：",
        ("🟢 綠幕模式 (推薦！)", "🤖 AI 模式 (白底用)")
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("📐 切割策略")
    slice_mode = st.sidebar.radio(
        "選擇切割方式：",
        ("🧠 智慧視覺偵測 (預設)", "📏 強制網格切割 (自訂行列)")
    )

    # 共用參數設定 (一次設定，套用到所有圖片)
    if slice_mode == "🧠 智慧視覺偵測 (預設)":
        dilation_size = st.sidebar.slider("膨脹係數 (防切字)", 5, 50, 25)
        st.sidebar.info("💡 適合：排列不規則，或背景很乾淨的圖。")
    else:
        st.sidebar.warning("⚠️ 強制模式：所有上傳的圖片必須是相同的網格排列。")
        col_r, col_c = st.sidebar.columns(2)
        with col_r:
            rows = st.number_input("縱向列數 (Rows)", min_value=1, value=5, step=1)
        with col_c:
            cols = st.number_input("橫向行數 (Columns)", min_value=1, value=6, step=1)

    # --- 綠幕演算法 ---
    def remove_green_screen_math(img_pil):
        img = np.array(img_pil.convert("RGBA"))
        r, g, b, a = img[:, :, 0], img[:, :, 1], img[:, :, 2], img[:, :, 3]
        mask = (g > 90) & (g > r + 15) & (g > b + 15)
        img[mask, 3] = 0
        return Image.fromarray(img)

    # --- 核心處理邏輯 (單張處理) ---
    def process_single_image(image_pil, mode_selection, slicing_strategy, dilation_val=25, grid_rows=5, grid_cols=6):
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
            # 排序：先由上到下，再由左到右
            bounding_boxes.sort(key=lambda x: (round(x[1]/100), x[0]))

            for x, y, w, h in bounding_boxes:
                sticker_cv = img_cv[y:y+h, x:x+w]
                sticker_pil = Image.fromarray(cv2.cvtColor(sticker_cv, cv2.COLOR_BGR2RGB))
                
                if "綠幕" in mode_selection:
                    sticker_no_bg = remove_green_screen_math(sticker_pil)
                else:
                    sticker_no_bg = remove(sticker_pil)
                
                bbox = sticker_no_bg.getbbox()
                if bbox:
                    sticker_trimmed = sticker_no_bg.crop(bbox)
                    sticker_final = sticker_trimmed.copy()
                    sticker_final.thumbnail((370, 320), Image.Resampling.LANCZOS)
                    processed_stickers.append(sticker_final)

        # 策略 B: 強制網格切割
        else:
            height, width, _ = img_cv.shape
            cell_h = height // grid_rows
            cell_w = width // grid_cols
            
            for r in range(grid_rows):
                for c in range(grid_cols):
                    x = c * cell_w
                    y = r * cell_h
                    
                    sticker_cv = img_cv[y:y+cell_h, x:x+cell_w]
                    sticker_pil = Image.fromarray(cv2.cvtColor(sticker_cv, cv2.COLOR_BGR2RGB))
                    
                    if "綠幕" in mode_selection:
                        sticker_no_bg = remove_green_screen_math(sticker_pil)
                    else:
                        sticker_no_bg = remove(sticker_pil)
                    
                    bbox = sticker_no_bg.getbbox()
                    if bbox:
                        sticker_trimmed = sticker_no_bg.crop(bbox)
                        sticker_final = sticker_trimmed.copy()
                        sticker_final.thumbnail((370, 320), Image.Resampling.LANCZOS)
                        processed_stickers.append(sticker_final)

        return processed_stickers

    # --- 主邏輯 UI (批量處理) ---
    if uploaded_files:
        # 顯示縮圖預覽
        st.write(f"📂 已選擇 {len(uploaded_files)} 個檔案")
        cols = st.columns(len(uploaded_files))
        for i, file in enumerate(uploaded_files):
            with cols[i]:
                st.image(file, caption=file.name, use_container_width=True)
        
        if st.button("🚀 批量開始處理！"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 準備 ZIP
            zip_buffer = io.BytesIO()
            total_stickers_count = 0
            
            # 設定參數
            d_val = dilation_size if "智慧" in slice_mode else 0
            r_val = rows if "強制" in slice_mode else 5
            c_val = cols if "強制" in slice_mode else 6
            
            try:
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    # 遍歷所有上傳的檔案
                    for idx, uploaded_file in enumerate(uploaded_files):
                        status_text.text(f"正在處理第 {idx+1}/{len(uploaded_files)} 張圖：{uploaded_file.name} ...")
                        
                        image = Image.open(uploaded_file).convert("RGB")
                        stickers = process_single_image(image, remove_mode, slice_mode, d_val, r_val, c_val)
                        
                        if not stickers:
                            st.warning(f"⚠️ 在 {uploaded_file.name} 中找不到貼圖，跳過。")
                            continue
                            
                        # 將切出來的貼圖寫入 ZIP，編號持續累加
                        for s in stickers:
                            total_stickers_count += 1
                            img_byte_arr = io.BytesIO()
                            s.save(img_byte_arr, format='PNG')
                            # 檔名格式：01.png, 02.png... 60.png
                            zf.writestr(f"{total_stickers_count:02d}.png", img_byte_arr.getvalue())
                        
                        # 更新進度條
                        progress_bar.progress((idx + 1) / len(uploaded_files))

                if total_stickers_count == 0:
                    st.error("⚠️ 所有圖片都處理失敗。")
                    st.stop()
                    
                st.success(f"✅ 全數完成！共產生 {total_stickers_count} 張貼圖，編號已自動排序 (01 ~ {total_stickers_count:02d})。")
                
                st.download_button(
                    label=f"📥 下載合併貼圖包 ({total_stickers_count}張)",
                    data=zip_buffer.getvalue(),
                    file_name="SarahDad_Batch_Stickers.zip",
                    mime="application/zip"
                )
                
            except Exception as e:
                st.error(f"錯誤: {e}")

# ==========================================
# 功能 B：製作主要與標籤圖片 (維持不變)
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
