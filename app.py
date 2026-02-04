import streamlit as st
from PIL import Image
from rembg import remove
import io
import zipfile
import numpy as np
import cv2

# --- 頁面設定 ---
st.set_page_config(page_title="莎拉爸貼圖神器 v6.9.1", page_icon="🐴", layout="wide")
st.title("🐴 莎拉爸貼圖神器 v6.9.1 (去背修復版)")
st.markdown("🚀 **v6.9.1 緊急修正**：放寬綠幕判定標準，解決去背失敗的問題！")

# --- Session State 初始化 ---
if 'processed_stickers' not in st.session_state:
    st.session_state.processed_stickers = []
if 'original_images' not in st.session_state:
    st.session_state.original_images = []
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

# --- 側邊欄：控制台 ---
st.sidebar.header("⚙️ 控制台")
if st.sidebar.button("🗑️ 清除重來 (Reset All)", type="secondary", use_container_width=True):
    st.session_state.processed_stickers = []
    st.session_state.original_images = []
    st.session_state.uploader_key += 1 
    st.rerun()
run_button = st.sidebar.button("🚀 開始處理圖片 (Start)", type="primary", use_container_width=True)

st.sidebar.markdown("---")

# --- 側邊欄：設定區 ---
st.sidebar.header("1. 上傳圖片")
uploaded_files = st.sidebar.file_uploader(
    "請上傳貼圖大圖 (可多選混搭)", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.uploader_key}"
)

st.sidebar.header("2. 去背與效果")
remove_mode = st.sidebar.radio(
    "選擇去背方式：",
    ("🟢 綠幕模式 (推薦！)", "🤖 AI 模式 (白底用)")
)
border_thickness = st.sidebar.slider("⚪ 白邊厚度 (0=無邊)", 0, 20, 8)

st.sidebar.header("3. 切割策略")
slice_mode = st.sidebar.radio(
    "選擇切割方式：",
    (
        "🧠 智慧視覺偵測 (不限格數)", 
        "🤖 強制網格 (自動判斷 6x5 / 8x5)", 
        "📏 強制網格 (手動設定)"
    )
)

manual_rows, manual_cols = 5, 6
dilation_size = 25

if "智慧" in slice_mode:
    dilation_size = st.sidebar.slider("膨脹係數 (防切字)", 5, 50, 25)
    st.sidebar.info("💡 適合：排列不規則，但間距必須足夠。")
elif "自動" in slice_mode:
    st.sidebar.success("✨ 程式將根據圖片長寬比，自動決定是用 6x5 還是 8x5 切割。")
else:
    st.sidebar.warning("⚠️ 手動模式：請自行設定行列數。")
    c1, c2 = st.sidebar.columns(2)
    with c1:
        manual_rows = st.number_input("縱向列數 (Rows)", 1, 10, 3) 
    with c2:
        manual_cols = st.number_input("橫向行數 (Cols)", 1, 10, 4) 

# --- 核心函數 ---

# [v6.9.1 修正] 放寬版綠幕去背算法
def remove_green_screen_math(img_pil):
    img = np.array(img_pil.convert("RGBA"))
    r, g, b, a = img[:, :, 0], img[:, :, 1], img[:, :, 2], img[:, :, 3]
    
    # --- 新的判斷邏輯 ---
    # 1. 綠色通道要夠亮 (>180)
    # 2. 綠色要比紅色高出至少 30
    # 3. 綠色要比藍色高出至少 30
    # 這個組合比舊版寬容，但比最早的版本嚴格，能有效平衡。
    mask = (g > 180) & (g > r + 30) & (g > b + 30)
    # --------------------
    
    img[mask, 3] = 0
    return Image.fromarray(img)

def add_white_border(image_pil, thickness):
    """為透明背景的圖片加上白色描邊"""
    if thickness == 0: return image_pil
    img = image_pil.convert("RGBA")
    alpha = img.getchannel('A')
    alpha_cv = np.array(alpha)
    kernel_size = thickness * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    border_mask_cv = cv2.dilate(alpha_cv, kernel, iterations=1)
    white_border_bg = Image.new("RGBA", img.size, (255, 255, 255, 0))
    white_border_bg.paste((255, 255, 255, 255), (0, 0), Image.fromarray(border_mask_cv))
    final_img = Image.alpha_composite(white_border_bg, img)
    return final_img

def process_single_image(image_pil, mode_selection, slicing_strategy, dilation_val, man_r, man_c, border_thick):
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    processed_stickers = []
    
    use_grid = False
    grid_rows, grid_cols = 5, 6
    
    if "智慧" in slicing_strategy: use_grid = False
    elif "手動" in slicing_strategy: use_grid = True; grid_rows, grid_cols = man_r, man_c
    elif "自動" in slicing_strategy:
        use_grid = True; h, w, _ = img_cv.shape; ratio = w / h
        if ratio > 1.4: grid_rows, grid_cols = 5, 8
        else: grid_rows, grid_cols = 5, 6

    # --- 內部的切割與後製流程 ---
    def post_process_sticker(sticker_pil_raw):
        # 1. 去背
        if "綠幕" in mode_selection:
            sticker_no_bg = remove_green_screen_math(sticker_pil_raw)
        else:
            sticker_no_bg = remove(sticker_pil_raw)
        
        # 2. 修剪透明邊緣 (Trim)
        bbox = sticker_no_bg.getbbox()
        if bbox:
            sticker_trimmed = sticker_no_bg.crop(bbox)
            
            # 3. 加上白邊
            sticker_with_border = add_white_border(sticker_trimmed, border_thick)
            
            # 4. 縮放與正規化
            sticker_final = sticker_with_border.copy()
            sticker_final.thumbnail((370, 320), Image.Resampling.LANCZOS)
            w_new, h_new = sticker_final.size
            if w_new % 2 != 0: w_new -= 1
            if h_new % 2 != 0: h_new -= 1
            if w_new != sticker_final.width or h_new != sticker_final.height:
                 sticker_final = sticker_final.resize((w_new, h_new), Image.Resampling.LANCZOS)
            return sticker_final
        return None

    # --- 執行切割 ---
    if not use_grid:
        # (智慧切割邏輯省略，與前版相同)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        if "綠幕" in mode_selection: _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else: _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
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
            final = post_process_sticker(sticker_pil)
            if final: processed_stickers.append(final)
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
                final = post_process_sticker(sticker_pil)
                if final: processed_stickers.append(final)

    return processed_stickers, (grid_rows, grid_cols) if use_grid else ("Smart", "Smart")

def create_resized_image(img, target_size):
    img = img.copy()
    img.thumbnail(target_size, Image.Resampling.LANCZOS)
    bg = Image.new("RGBA", target_size, (0, 0, 0, 0))
    left = (target_size[0] - img.width) // 2
    top = (target_size[1] - img.height) // 2
    bg.paste(img, (left, top))
    return bg

# --- 主程式區 ---
if run_button:
    if not uploaded_files:
        st.error("⚠️ 請先上傳圖片再按開始！")
    else:
        st.session_state.processed_stickers = []
        st.session_state.original_images = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        try:
            for idx, uploaded_file in enumerate(uploaded_files):
                image = Image.open(uploaded_file).convert("RGB")
                st.session_state.original_images.append((uploaded_file.name, image))
                stickers, strategy_used = process_single_image(
                    image, remove_mode, slice_mode, dilation_size, manual_rows, manual_cols, border_thickness
                )
                status_text.text(f"正在處理：{uploaded_file.name} ...")
                st.session_state.processed_stickers.extend(stickers)
                progress_bar.progress((idx + 1) / len(uploaded_files))
            if not st.session_state.processed_stickers: st.error("⚠️ 未偵測到貼圖。")
            else: st.success(f"✅ 完成！共 {len(st.session_state.processed_stickers)} 張。")
        except Exception as e: st.error(f"錯誤: {e}")

# --- 預覽與下載區 (保持不變) ---
if st.session_state.processed_stickers:
    st.divider()
    st.header("🖼️ 貼圖總覽")
    total_stickers = len(st.session_state.processed_stickers)
