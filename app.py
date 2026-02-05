import streamlit as st
from PIL import Image
from rembg import remove
import io
import zipfile
import numpy as np
import cv2

# --- 頁面設定 ---
st.set_page_config(page_title="莎拉爸貼圖神器 v6.4", page_icon="🐴", layout="wide")
st.title("🐴 莎拉爸貼圖神器 v6.4 (LINE 規格修正版)")
st.markdown("🚀 **v6.4 更新**：強制輸出標準 **370x320 (偶數)** 尺寸，解決 LINE 上架報錯問題。")

# --- Session State ---
if 'processed_stickers' not in st.session_state:
    st.session_state.processed_stickers = []
if 'original_images' not in st.session_state:
    st.session_state.original_images = []

# --- 側邊欄：設定區 ---
st.sidebar.header("⚙️ 1. 參數設定")

uploaded_files = st.sidebar.file_uploader(
    "請上傳貼圖大圖 (可多選混搭)", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

st.sidebar.header("🎨 2. 去背模式")
remove_mode = st.sidebar.radio(
    "選擇去背方式：",
    ("🟢 綠幕模式 (推薦！)", "🤖 AI 模式 (白底用)")
)

st.sidebar.header("📐 3. 切割策略")
slice_mode = st.sidebar.radio(
    "選擇切割方式：",
    (
        "🧠 智慧視覺偵測 (不限格數)", 
        "🤖 強制網格 (自動判斷 6x5 / 8x5)", 
        "📏 強制網格 (手動設定)"
    )
)

# 參數顯示邏輯
manual_rows, manual_cols = 5, 6
dilation_size = 25

if "智慧" in slice_mode:
    dilation_size = st.sidebar.slider("膨脹係數 (防切字)", 5, 50, 25)
    st.sidebar.info("💡 適合：排列不規則，但間距必須足夠。")
elif "自動" in slice_mode:
    st.sidebar.success("✨ 程式將根據圖片長寬比，自動決定是用 6x5 還是 8x5 切割。")
else:
    st.sidebar.warning("⚠️ 手動模式：4x3 大圖請手動設為 Rows=3, Cols=4")
    c1, c2 = st.sidebar.columns(2)
    with c1:
        manual_rows = st.number_input("縱向列數 (Rows)", 1, 10, 5)
    with c2:
        manual_cols = st.number_input("橫向行數 (Cols)", 1, 10, 6)

# --- 核心函數 ---

def remove_green_screen_math(img_pil):
    img = np.array(img_pil.convert("RGBA"))
    r, g, b, a = img[:, :, 0], img[:, :, 1], img[:, :, 2], img[:, :, 3]
    mask = (g > 90) & (g > r + 15) & (g > b + 15)
    img[mask, 3] = 0
    return Image.fromarray(img)

def process_single_image(image_pil, mode_selection, slicing_strategy, dilation_val=25, man_r=5, man_c=6):
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    processed_stickers = []
    
    # 決定網格數 (Rows, Cols)
    use_grid = False
    grid_rows, grid_cols = 5, 6
    
    if "智慧" in slicing_strategy:
        use_grid = False
    elif "手動" in slicing_strategy:
        use_grid = True
        grid_rows, grid_cols = man_r, man_c
    elif "自動" in slicing_strategy:
        use_grid = True
        # v6.3 的核心判斷邏輯
        h, w, _ = img_cv.shape
        ratio = w / h
        if ratio > 1.4: 
            grid_rows, grid_cols = 5, 8
        else:
            grid_rows, grid_cols = 5, 6

    # --- v6.4 核心修正：定義統一的處理函式 ---
    # 這個函式負責：去背 -> 裁切 -> 【強制補成 370x320 偶數畫布】
    def extract_and_resize(sticker_img_pil):
        # 1. 去背
        if "綠幕" in mode_selection:
            sticker_no_bg = remove_green_screen_math(sticker_img_pil)
        else:
            sticker_no_bg = remove(sticker_img_pil)
        
        # 2. 裁切多餘白邊
        bbox = sticker_no_bg.getbbox()
        if bbox:
            sticker_cropped = sticker_no_bg.crop(bbox)
            
            # 3. 縮放限制 (保持比例)
            sticker_cropped.thumbnail((370, 320), Image.Resampling.LANCZOS)
            
            # 4. 【關鍵修正】建立標準偶數畫布 (370x320)
            final_bg = Image.new("RGBA", (370, 320), (0, 0, 0, 0))
            
            # 計算置中
            left = (370 - sticker_cropped.width) // 2
            top = (320 - sticker_cropped.height) // 2
            
            # 貼上
            final_bg.paste(sticker_cropped, (left, top))
            return final_bg
        return None

    # --- 執行切割 ---
    if not use_grid:
        # 智慧視覺邏輯
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
            # 呼叫上面的修正函式
            result = extract_and_resize(sticker_pil)
            if result: processed_stickers.append(result)
    
    else:
        # 強制網格邏輯
        height, width, _ = img_cv.shape
        cell_h = height // grid_rows
        cell_w = width // grid_cols
        
        for r in range(grid_rows):
            for c in range(grid_cols):
                x = c * cell_w
                y = r * cell_h
                sticker_cv = img_cv[y:y+cell_h, x:x+cell_w]
                sticker_pil = Image.fromarray(cv2.cvtColor(sticker_cv, cv2.COLOR_BGR2RGB))
                # 呼叫上面的修正函式
                result = extract_and_resize(sticker_pil)
                if result: processed_stickers.append(result)

    return processed_stickers, (grid_rows, grid_cols) if use_grid else ("Smart", "Smart")

def create_resized_image(img, target_size):
    # Main/Tab 也要確保偶數，雖然通常設定值就是偶數
    img = img.copy()
    img.thumbnail(target_size, Image.Resampling.LANCZOS)
    bg = Image.new("RGBA", target_size, (0, 0, 0, 0))
    left = (target_size[0] - img.width) // 2
    top = (target_size[1] - img.height) // 2
    bg.paste(img, (left, top))
    return bg

# --- 主程式 ---

if uploaded_files:
    if st.sidebar.button("🚀 開始處理圖片"):
        st.session_state.processed_stickers = []
        st.session_state.original_images = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            for idx, uploaded_file in enumerate(uploaded_files):
                image = Image.open(uploaded_file).convert("RGB")
                st.session_state.original_images.append((uploaded_file.name, image))
                
                # 執行處理
                stickers, strategy_used = process_single_image(
                    image, remove_mode, slice_mode, dilation_size, manual_rows, manual_cols
                )
                
                if "自動" in slice_mode:
                    status_text.text(f"正在處理：{uploaded_file.name} (偵測為 {strategy_used[1]}x{strategy_used[0]} 網格)...")
                else:
                    status_text.text(f"正在處理：{uploaded_file.name} ...")
                
                st.session_state.processed_stickers.extend(stickers)
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            if not st.session_state.processed_stickers:
                st.error("⚠️ 未偵測到貼圖。")
            else:
                st.success(f"✅ 完成！共 {len(st.session_state.processed_stickers)} 張。(已自動修正為偶數尺寸)")
                
        except Exception as e:
            st.error(f"錯誤: {e}")

# --- 預覽與下載區 ---
if st.session_state.processed_stickers:
    st.divider()
    st.header("🖼️ 貼圖總覽與設定")
    
    total_stickers = len(st.session_state.processed_stickers)
    sticker_options = [f"{i+1:02d}" for i in range(total_stickers)]
    
    col_selectors, col_preview = st.columns([1, 2])
    
    with col_selectors:
        st.subheader("設定關鍵圖片")
        main_idx = int(st.selectbox("⭐ Main 圖片", sticker_options, index=0)) - 1
        tab_idx = int(st.selectbox("🏷️ Tab 圖片", sticker_options, index=0)) - 1
        
        main_img = create_resized_image(st.session_state.processed_stickers[main_idx], (240, 240))
        tab_img = create_resized_image(st.session_state.processed_stickers[tab_idx], (96, 74))
        
        c1, c2 = st.columns(2)
        c1.image(main_img, caption="Main")
        c2.image(tab_img, caption="Tab")

    with col_preview:
        st.subheader("全部預覽")
        preview_cols = st.columns(6)
        for i, sticker in enumerate(st.session_state.processed_stickers):
            with preview_cols[i % 6]:
                st.image(sticker, caption=f"{i+1:02d}", use_container_width=True)

    st.divider()
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        # 下載原圖
        for name, img in st.session_state.original_images:
            img_byte = io.BytesIO()
            img.save(img_byte, format='PNG')
            zf.writestr(f"Originals/{name.replace('.jpg','.png')}", img_byte.getvalue())
        # 下載貼圖
        for i, sticker in enumerate(st.session_state.processed_stickers):
            sticker_byte = io.BytesIO()
            sticker.save(sticker_byte, format='PNG')
            zf.writestr(f"Stickers/{i+1:02d}.png", sticker_byte.getvalue())
        
        # 下載 Main/Tab
        main_byte = io.BytesIO()
        main_img.save(main_byte, format='PNG')
        zf.writestr("main.png", main_byte.getvalue())
        tab_byte = io.BytesIO()
        tab_img.save(tab_byte, format='PNG')
        zf.writestr("tab.png", tab_byte.getvalue())

    st.download_button(
        label=f"📦 下載修正版懶人包 (偶數尺寸)",
        data=zip_buffer.getvalue(),
        file_name="SarahDad_Fixed_Stickers.zip",
        mime="application/zip",
        type="primary"
    )
