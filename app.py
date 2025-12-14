import streamlit as st
from PIL import Image
from rembg import remove
import io
import zipfile
import numpy as np
import cv2

# --- 頁面設定 ---
st.set_page_config(page_title="莎拉爸貼圖神器 v6.2", page_icon="🐴", layout="wide")
st.title("🐴 莎拉爸貼圖神器 v6.2 (全流程整合版)")
st.markdown("🚀 **流程優化**：上傳大圖 > 自動切割 > 線上預覽挑選 Main/Tab > 一鍵打包下載全部！")

# --- Session State 初始化 (關鍵：用來記住切好的圖) ---
if 'processed_stickers' not in st.session_state:
    st.session_state.processed_stickers = []
if 'original_images' not in st.session_state:
    st.session_state.original_images = []

# --- 側邊欄：設定區 ---
st.sidebar.header("⚙️ 1. 參數設定")

# 檔案上傳
uploaded_files = st.sidebar.file_uploader(
    "請上傳貼圖大圖 (可多選)", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

# 去背模式
st.sidebar.header("🎨 2. 去背模式")
remove_mode = st.sidebar.radio(
    "選擇去背方式：",
    ("🟢 綠幕模式 (推薦！)", "🤖 AI 模式 (白底用)")
)

# 切割策略
st.sidebar.header("📐 3. 切割策略")
slice_mode = st.sidebar.radio(
    "選擇切割方式：",
    ("🧠 智慧視覺偵測 (預設)", "📏 強制網格切割 (自訂行列)")
)

if slice_mode == "🧠 智慧視覺偵測 (預設)":
    dilation_size = st.sidebar.slider("膨脹係數 (防切字)", 5, 50, 25)
    st.sidebar.info("💡 適合：排列不規則，或背景很乾淨的圖。")
    rows, cols = 5, 6 # 預設值，雖不使用但避免變數未定義
else:
    st.sidebar.warning("⚠️ 強制模式：所有圖片需有相同網格。")
    c1, c2 = st.sidebar.columns(2)
    with c1:
        rows = st.number_input("縱向列數 (Rows)", 1, 10, 5)
    with c2:
        cols = st.number_input("橫向行數 (Cols)", 1, 10, 6)

# --- 核心函數區 ---

def remove_green_screen_math(img_pil):
    img = np.array(img_pil.convert("RGBA"))
    r, g, b, a = img[:, :, 0], img[:, :, 1], img[:, :, 2], img[:, :, 3]
    # 綠幕去背邏輯
    mask = (g > 90) & (g > r + 15) & (g > b + 15)
    img[mask, 3] = 0
    return Image.fromarray(img)

def process_single_image(image_pil, mode_selection, slicing_strategy, dilation_val=25, grid_rows=5, grid_cols=6):
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    processed_stickers = []
    
    # --- 策略 A: 智慧視覺 ---
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
            sticker_cv = img_cv[y:y+h, x:x+w]
            sticker_pil = Image.fromarray(cv2.cvtColor(sticker_cv, cv2.COLOR_BGR2RGB))
            
            if "綠幕" in mode_selection:
                sticker_no_bg = remove_green_screen_math(sticker_pil)
            else:
                sticker_no_bg = remove(sticker_pil)
            
            bbox = sticker_no_bg.getbbox()
            if bbox:
                sticker_final = sticker_no_bg.crop(bbox)
                sticker_final.thumbnail((370, 320), Image.Resampling.LANCZOS)
                processed_stickers.append(sticker_final)

    # --- 策略 B: 強制網格 ---
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
                    sticker_final = sticker_no_bg.crop(bbox)
                    sticker_final.thumbnail((370, 320), Image.Resampling.LANCZOS)
                    processed_stickers.append(sticker_final)

    return processed_stickers

def create_resized_image(img, target_size):
    """將圖片縮放並居中放置在透明背景上"""
    img = img.copy()
    img.thumbnail(target_size, Image.Resampling.LANCZOS)
    bg = Image.new("RGBA", target_size, (0, 0, 0, 0))
    # 居中計算
    left = (target_size[0] - img.width) // 2
    top = (target_size[1] - img.height) // 2
    bg.paste(img, (left, top))
    return bg

# --- 主程式邏輯 ---

# 1. 按鈕區：開始處理
if uploaded_files:
    if st.sidebar.button("🚀 開始處理圖片"):
        # 清空舊資料
        st.session_state.processed_stickers = []
        st.session_state.original_images = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 參數準備
        d_val = dilation_size if "智慧" in slice_mode else 0
        r_val = rows if "強制" in slice_mode else 5
        c_val = cols if "強制" in slice_mode else 6
        
        try:
            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"正在處理大圖：{uploaded_file.name} ...")
                
                # 讀取並儲存原圖 (供下載用)
                image = Image.open(uploaded_file).convert("RGB")
                st.session_state.original_images.append((uploaded_file.name, image))
                
                # 切割處理
                stickers = process_single_image(image, remove_mode, slice_mode, d_val, r_val, c_val)
                st.session_state.processed_stickers.extend(stickers)
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            if not st.session_state.processed_stickers:
                st.error("⚠️ 未偵測到任何貼圖，請檢查設定。")
            else:
                st.success(f"✅ 處理完成！共獲得 {len(st.session_state.processed_stickers)} 張貼圖。請在下方挑選 Main/Tab。")
                
        except Exception as e:
            st.error(f"錯誤: {e}")

# 2. 預覽與選取區 (只有在有資料時顯示)
if st.session_state.processed_stickers:
    st.divider()
    st.header("🖼️ 貼圖總覽與設定")
    
    # 顯示所有貼圖的縮圖 (低解析度預覽，實際上不降解析度，Streamlit 會自動縮圖顯示，但我們可以控制 column 寬度)
    total_stickers = len(st.session_state.processed_stickers)
    
    # 建立選項清單 (例如: "01", "02"...)
    sticker_options = [f"{i+1:02d}" for i in range(total_stickers)]
    
    # --- 挑選 Main 與 Tab ---
    col_selectors, col_preview = st.columns([1, 2])
    
    with col_selectors:
        st.subheader("設定關鍵圖片")
        st.info("請從右側預覽圖中，記下喜歡的貼圖編號。")
        
        # 選擇 Main
        main_idx_str = st.selectbox("⭐ 選擇 Main 圖片 (主要圖片)", sticker_options, index=0)
        main_idx = int(main_idx_str) - 1
        
        # 選擇 Tab
        tab_idx_str = st.selectbox("🏷️ 選擇 Tab 圖片 (標籤圖片)", sticker_options, index=0)
        tab_idx = int(tab_idx_str) - 1
        
        # 即時生成預覽
        main_img = create_resized_image(st.session_state.processed_stickers[main_idx], (240, 240))
        tab_img = create_resized_image(st.session_state.processed_stickers[tab_idx], (96, 74))
        
        # 顯示 Main/Tab 預覽
        p1, p2 = st.columns(2)
        with p1:
            st.image(main_img, caption="Main (240x240)")
        with p2:
            st.image(tab_img, caption="Tab (96x74)")

    with col_preview:
        st.subheader("全部表情預覽")
        # 顯示網格
        preview_cols = st.columns(6) # 6欄顯示
        for i, sticker in enumerate(st.session_state.processed_stickers):
            with preview_cols[i % 6]:
                st.image(sticker, caption=f"{i+1:02d}", use_container_width=True)

    # 3. 下載區
    st.divider()
    st.subheader("📥 打包下載")
    
    # 準備 ZIP
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        # A. 寫入原圖 (Original Big Images)
        for name, img in st.session_state.original_images:
            img_byte = io.BytesIO()
            img.save(img_byte, format='PNG')
            zf.writestr(f"Originals/{name.replace('.jpg','.png')}", img_byte.getvalue())
            
        # B. 寫入所有切好的貼圖 (Stickers)
        for i, sticker in enumerate(st.session_state.processed_stickers):
            sticker_byte = io.BytesIO()
            sticker.save(sticker_byte, format='PNG')
            zf.writestr(f"Stickers/{i+1:02d}.png", sticker_byte.getvalue())
            
        # C. 寫入 Main 與 Tab
        main_byte = io.BytesIO()
        main_img.save(main_byte, format='PNG')
        zf.writestr("main.png", main_byte.getvalue())
        
        tab_byte = io.BytesIO()
        tab_img.save(tab_byte, format='PNG')
        zf.writestr("tab.png", tab_byte.getvalue())

    st.download_button(
        label=f"📦 下載完整懶人包 (含 {total_stickers} 張貼圖 + 原圖 + Main/Tab)",
        data=zip_buffer.getvalue(),
        file_name="SarahDad_Full_Package.zip",
        mime="application/zip",
        type="primary" # 讓按鈕變顯眼
    )

else:
    st.info("👈 請先在左側上傳圖片並點擊「開始處理」。")
