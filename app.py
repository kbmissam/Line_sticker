import streamlit as st
from PIL import Image, ImageDraw, ImageOps, ImageFilter
from rembg import remove
import io
import zipfile
import numpy as np
import cv2

# --- 頁面設定 ---
st.set_page_config(page_title="莎拉爸貼圖神器 v7.3", page_icon="🐴", layout="wide")
st.title("🐴 莎拉爸貼圖神器 v7.3 (語法修復版)")
st.markdown("🚀 **v7.3 更新**：修復程式碼語法錯誤，保留 v7.2 所有強大功能 (亮部保護+敏感度調節+白邊效果)。")

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

# 這裡定義按鈕，但邏輯會在最下方執行
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
highlight_protection = 30 
border_thickness = 8

if "綠幕" in remove_mode:
    st.sidebar.markdown("##### 🔧 去背微調 (修復破洞)")
    
    gs_sensitivity = st.sidebar.slider(
        "🟢 綠幕敏感度 (Sensitivity)", 
        min_value=0, max_value=100, value=50, 
        help="【數值越小越安全】。如果角色破洞，請嘗試「調低」此數值。"
    )
    
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

# 1. 綠幕去背核心 (補全版)
def remove_green_screen_hsv(img_pil, sensitivity=50, white_protect=30):
    img = np.array(img_pil.convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # A. 建立綠幕遮罩
    # sensitivity 越高，對綠色的容忍度越高 (越容易把非綠色切掉)
    # 這裡做一個反向映射：使用者拉 50 -> 門檻 95
    sat_threshold = 140 - int(sensitivity * 0.9) 
    lower_green = np.array([35, sat_threshold, 40])
    upper_green = np.array([85, 255, 255])
    
    # 這是「綠色區域」的遮罩 (白色=綠色背景)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # B. 建立亮部保護遮罩
    if white_protect > 0:
        # 參數映射：保護值越高，對「白」的定義越寬鬆
        protect_s_max = int(white_protect * 0.8) 
        protect_v_min = 255 - int(white_protect * 1.5) 
        
        lower_white = np.array([0, 0, protect_v_min])      
        upper_white = np.array([180, protect_s_max, 255]) 
        
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # 關鍵邏輯：如果是綠色背景(green_mask)，但同時又是亮部(white_mask)，
        # 我們要把這些區域從「綠色背景」中移除 -> 視為前景
        final_green_mask = cv2.bitwise_and(green_mask, cv2.bitwise_not(white_mask))
    else:
        final_green_mask = green_mask

    # C. 應用遮罩去背
    # 將圖片轉為 RGBA
    img_rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
    
    # 遮罩為白色(255)的地方是背景，設為全透明
    img_rgba[final_green_mask > 0] = (0, 0, 0, 0)
    
    return Image.fromarray(cv2.cvtColor(img_rgba, cv2.COLOR_BGRA2RGBA))

# 2. 加白邊函數
def add_stroke(img_pil, thickness=8, color=(255, 255, 255, 255)):
    if thickness == 0:
        return img_pil
    
    img = img_pil.convert("RGBA")
    # 取得 Alpha 通道
    r, g, b, a = img.split()
    
    # 擴張邊緣 (Dilation) 來製作外框
    # 先將 Alpha 轉為 numpy array
    alpha_np = np.array(a)
    kernel = np.ones((thickness * 2 + 1, thickness * 2 + 1), np.uint8)
    
    # 使用 cv2.dilate 讓不透明區域變胖
    outline_alpha = cv2.dilate(alpha_np, kernel, iterations=1)
    
    # 建立純色背景
    stroke_bg = Image.new("RGBA", img.size, color)
    stroke_bg.putalpha(Image.fromarray(outline_alpha))
    
    # 將原圖疊在白邊圖之上
    final_img = Image.alpha_composite(stroke_bg, img)
    return final_img

# 3. 圖片處理整合函數 (去背 -> 加框 -> 裁切 -> 補正尺寸)
def extract_and_resize(sticker_img_pil, mode_selection, sensitivity, protect, border):
    # 1. 去背
    if "綠幕" in mode_selection:
        sticker_no_bg = remove_green_screen_hsv(sticker_img_pil, sensitivity, protect)
    else:
        sticker_no_bg = remove(sticker_img_pil)
    
    # 2. 裁切多餘白邊 (Crop)
    bbox = sticker_no_bg.getbbox()
    if bbox:
        sticker_cropped = sticker_no_bg.crop(bbox)
        
        # 3. 加白邊 (如果有設定)
        if border > 0:
            sticker_cropped = add_stroke(sticker_cropped, border)
            # 加框後可能會有新的邊界，可以再次 crop 或保持原樣
            # 為了保險，這裡就不再 crop，直接縮放

        # 4. 縮放限制 (保持比例縮小，直到長寬都 <= 370x320)
        sticker_cropped.thumbnail((370, 320), Image.Resampling.LANCZOS)
        
        # 5. 【關鍵修正】建立標準偶數畫布 (370x320)
        final_bg = Image.new("RGBA", (370, 320), (0, 0, 0, 0))
        
        # 計算置中位置
        left = (370 - sticker_cropped.width) // 2
        top = (320 - sticker_cropped.height) // 2
        
        # 貼上
        final_bg.paste(sticker_cropped, (left, top))
        return final_bg
    return None

def process_single_image(image_pil, mode_selection, slicing_strategy, dilation_val=25, man_r=5, man_c=6, sensitivity=50, protect=30, border=8):
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    processed_stickers = []
    
    use_grid = False
    grid_rows, grid_cols = 5, 6
    
    if "智慧" in slicing_strategy:
        use_grid = False
    elif "手動" in slicing_strategy:
        use_grid = True
        grid_rows, grid_cols = man_r, man_c
    elif "自動" in slicing_strategy:
        use_grid = True
        h, w, _ = img_cv.shape
        ratio = w / h
        if ratio > 1.4: 
            grid_rows, grid_cols = 5, 8
        else:
            grid_rows, grid_cols = 5, 6

    # --- 執行切割 ---
    if not use_grid:
        # 智慧視覺邏輯
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # 簡單閾值做輪廓偵測 (不影響最後去背，只為了切開)
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
            # 呼叫整合處理
            result = extract_and_resize(sticker_pil, mode_selection, sensitivity, protect, border)
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
                # 呼叫整合處理
                result = extract_and_resize(sticker_pil, mode_selection, sensitivity, protect, border)
                if result: processed_stickers.append(result)

    return processed_stickers, (grid_rows, grid_cols) if use_grid else ("Smart", "Smart")

def create_resized_image(img, target_size):
    img = img.copy()
    img.thumbnail(target_size, Image.Resampling.LANCZOS)
    bg = Image.new("RGBA", target_size, (0, 0, 0, 0))
    left = (target_size[0] - img.width) // 2
    top = (target_size[1] - img.height) // 2
    bg.paste(img, (left, top))
    return bg

# --- 主程式邏輯 (被按鈕觸發) ---

if run_button:
    if not uploaded_files:
        st.error("❌ 請先上傳圖片！")
    else:
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
                    image, remove_mode, slice_mode, dilation_size, 
                    manual_rows, manual_cols,
                    gs_sensitivity, highlight_protection, border_thickness
                )
                
                info_msg = f"正在處理：{uploaded_file.name}"
                if "自動" in slice_mode:
                    info_msg += f" (偵測為 {strategy_used[1]}x{strategy_used[0]} 網格)"
                status_text.text(f"{info_msg}...")
                
                st.session_state.processed_stickers.extend(stickers)
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            if not st.session_state.processed_stickers:
                st.error("⚠️ 未偵測到貼圖，請檢查切割策略或去背設定。")
            else:
                st.success(f"✅ 完成！共 {len(st.session_state.processed_stickers)} 張。")
                
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
        # 計算每行顯示數量
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
        label=f"📦 下載 v7.3 完整懶人包",
        data=zip_buffer.getvalue(),
        file_name="SarahDad_v7.3_Stickers.zip",
        mime="application/zip",
        type="primary"
    )
