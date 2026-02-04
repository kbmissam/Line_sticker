import streamlit as st
from PIL import Image, ImageDraw
from rembg import remove
import io
import zipfile
import numpy as np
import cv2

# --- 頁面設定 ---
st.set_page_config(page_title="莎拉爸貼圖神器 v7.2", page_icon="🐴", layout="wide")
st.title("🐴 莎拉爸貼圖神器 v7.2 (亮部守護版)")
st.markdown("🚀 **v7.2 更新**：新增「亮部保護」功能，專門修復角色反光破洞問題！")

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

st.sidebar.header("2. 去背與修復")
remove_mode = st.sidebar.radio(
    "選擇去背方式：",
    ("🟢 綠幕模式 (專家微調)", "🤖 AI 模式 (白底用)")
)

# 預設參數
gs_sensitivity = 50
highlight_protection = 30 # 預設開啟保護
border_thickness = 8

if "綠幕" in remove_mode:
    st.sidebar.markdown("##### 🔧 去背微調 (修復破洞)")
    
    # 1. 綠幕敏感度
    gs_sensitivity = st.sidebar.slider(
        "🟢 綠幕敏感度 (Sensitivity)", 
        min_value=0, max_value=100, value=50, 
        help="【數值越小越安全】。如果角色破洞，請嘗試「調低」此數值。"
    )
    
    # 2. [v7.2 新增] 亮部保護
    highlight_protection = st.sidebar.slider(
        "💡 亮部保護 (White Protection)", 
        min_value=0, max_value=100, value=30, 
        help="【專修頭頂破洞】數值越高，越強行保留白色的部分。如果反光處被切掉，請「調高」此數值。"
    )

st.sidebar.markdown("##### ✨ 裝飾設定")
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

# [v7.2 核心] HSV 去背 + 亮部保護
def remove_green_screen_hsv(img_pil, sensitivity=50, white_protect=30):
    img = np.array(img_pil.convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # --- 1. 建立綠幕遮罩 (Green Mask) ---
    # Sensitivity (0-100) -> Saturation Threshold (150 - 50)
    # Sens=100 (Strict) -> Threshold=50 (Delete faint green)
    # Sens=0 (Loose) -> Threshold=150 (Only delete super green)
    
    # 修正邏輯：敏感度越高，門檻越低 (越容易被當成背景)
    sat_threshold = 140 - int(sensitivity * 0.9) # 50->95, 100->50, 0->140
    
    # H: 綠色中心約 60。範圍寬度固定為 +/- 25
    lower_green = np.array([35, sat_threshold, 40])
    upper_green = np.array([85, 255, 255])
    
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # --- 2. 建立亮部保護遮罩 (Highlight Mask) ---
    # 保護邏輯：飽和度很低 (接近白/灰) 且 亮度很高 (很亮)
    # White Protect (0-100) -> 調控對「白色」的寬容度
    
    if white_protect > 0:
        # S上限：保護程度越高，允許越高的飽和度被視為白色 (max 60)
        # V下限：保護程度越高，允許越暗的顏色被視為白色 (min 150)
        protect_s_max = int(white_protect * 0.8) # 30 -> 24
        protect_v_min = 255 - int(white_protect * 1.5) # 30 -> 210
        
        # 嚴格定義「白色/反光」
        lower_white = np.array([0, 0, protect_v_min])     # 亮度要夠
        upper_white = np.array([180, protect_s_max, 255]) # 飽和度要低
        
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # --- 3. 運算：綠幕 - 保護區 ---
        # 從綠幕遮罩中，挖掉屬於白色的部分
        final_mask = cv2.bitwise_and(green_mask, cv2.bitwise_not(white_mask))
    else:
        final_mask = green_mask
    
    # --- 4. 應用遮罩 ---
    mask_inv = cv2.bitwise_not(final_mask)
    img_rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGBA)
    img_rgba[:, :, 3] = mask_inv
    
    return Image.fromarray(img_rgba)

def add_white_border(image_pil, thickness):
    if thickness == 0: return image_pil
    try:
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
    except Exception:
        return image_pil

def create_checkerboard_bg(size, check_size=20):
    w, h = size
    img = Image.new("RGBA", (w, h), (220, 220, 220, 255))
    draw = ImageDraw.Draw(img)
    for x in range(0, w, check_size):
        for y in range(0, h, check_size):
            if (x // check_size + y // check_size) % 2 == 0:
                draw.rectangle([x, y, x + check_size, y + check_size], fill=(255, 255, 255, 255))
    return img

def make_preview(img_pil):
    bg = create_checkerboard_bg(img_pil.size, check_size=20)
    return Image.alpha_composite(bg, img_pil.convert("RGBA"))

def process_single_image(image_pil, mode_selection, slicing_strategy, dilation_val, man_r, man_c, border_thick, gs_sens, white_prot):
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

    def post_process_sticker(sticker_pil_raw):
        if "綠幕" in mode_selection:
            # [v7.2] 傳入兩個參數：敏感度 + 亮部保護
            sticker_no_bg = remove_green_screen_hsv(sticker_pil_raw, gs_sens, white_prot)
        else:
            sticker_no_bg = remove(sticker_pil_raw)
        
        bbox = sticker_no_bg.getbbox()
        if bbox:
            sticker_trimmed = sticker_no_bg.crop(bbox)
            sticker_with_border = add_white_border(sticker_trimmed, border_thick)
            
            sticker_final = sticker_with_border.copy()
            sticker_final.thumbnail((370, 320), Image.Resampling.LANCZOS)
            w_new, h_new = sticker_final.size
            if w_new % 2 != 0: w_new -= 1
            if h_new % 2 != 0: h_new -= 1
            if w_new == 0 or h_new == 0: return None
            
            if w_new != sticker_final.width or h_new != sticker_final.height:
                 sticker_final = sticker_final.resize((w_new, h_new), Image.Resampling.LANCZOS)
            return sticker_final
        return None

    if not use_grid:
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
                
                # [v7.2] 傳入 highlight_protection
                stickers, strategy_used = process_single_image(
                    image, remove_mode, slice_mode, dilation_size, 
                    manual_rows, manual_cols, border_thickness, gs_sensitivity, highlight_protection
                )
                
                status_text.text(f"正在處理：{uploaded_file.name} ...")
                st.session_state.processed_stickers.extend(stickers)
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            if not st.session_state.processed_stickers:
                st.error("⚠️ 未偵測到貼圖。請調整去背參數。")
            else:
                st.success(f"✅ 完成！共 {len(st.session_state.processed_stickers)} 張。")
                
        except Exception as e:
            st.error(f"處理過程發生錯誤: {e}")

# --- 預覽與下載區 ---
if st.session_state.processed_stickers:
    st.divider()
    st.header("🖼️ 貼圖總覽 (檢視是否有破洞)")
    
    try:
        total_stickers = len(st.session_state.processed_stickers)
        sticker_options = [f"{i+1:02d}" for i in range(total_stickers)]
        
        col_selectors, col_preview = st.columns([1, 2])
        
        with col_selectors:
            st.subheader("設定 Main/Tab")
            if sticker_options:
                main_idx = int(st.selectbox("⭐ Main 圖片", sticker_options
