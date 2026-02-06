import streamlit as st
from PIL import Image, ImageDraw, ImageOps, ImageFilter
from rembg import remove
import io
import zipfile
import numpy as np
import cv2

# --- 頁面設定 ---
st.set_page_config(page_title="莎拉爸貼圖神器 v10.0", page_icon="🐴", layout="wide")
st.title("🐴 莎拉爸貼圖神器 v10.0 (智慧綠幕修復版)")
st.markdown("🚀 **v10.0 更新**：重寫「智慧視覺」演算法，現在能精準識別綠幕背景，解決「整張不切」的問題。")

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

# 執行按鈕 (放在側邊欄下方)
run_button = st.sidebar.button("🚀 開始處理圖片 (Start)", type="primary", use_container_width=True)

st.sidebar.markdown("---")

# --- 側邊欄：設定區 ---
st.sidebar.header("1. 上傳圖片")
uploaded_files = st.sidebar.file_uploader(
    "請上傳貼圖大圖", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.uploader_key}"
)

st.sidebar.header("2. 去背與修復")
remove_mode = st.sidebar.radio(
    "選擇去背方式：",
    ("🟢 綠幕模式 (專家微調)", "🤖 AI 模式 (白底用)")
)

gs_sensitivity = 50
highlight_protection = 30 
border_thickness = 8

if "綠幕" in remove_mode:
    st.sidebar.markdown("##### 🔧 去背微調")
    gs_sensitivity = st.sidebar.slider("🟢 綠幕敏感度", 0, 100, 50)
    highlight_protection = st.sidebar.slider("💡 亮部保護", 0, 100, 30)

st.sidebar.markdown("##### ✨ 裝飾與修整")
border_thickness = st.sidebar.slider("⚪ 白邊厚度", 0, 20, 8)
edge_crop = st.sidebar.slider("✂️ 邊緣內縮 (Edge Crop)", 0, 20, 0)

st.sidebar.header("3. 切割策略")
slice_mode = st.sidebar.radio(
    "選擇切割方式：",
    (
        "🧠 智慧視覺偵測 (推薦)", 
        "📏 自由多線微調", 
        "🤖 強制網格 (平均分配)"
    )
)

# --- 核心變數與滑桿 ---
off_v1, off_v2, off_v3 = 0, 0, 0
off_h1, off_h2 = 0, 0
dilation_size = 25

if "智慧" in slice_mode:
    st.sidebar.markdown("##### 🧠 智慧偵測設定")
    dilation_size = st.sidebar.slider("膨脹係數 (黏合力)", 5, 100, 30, 
                                      help="數值越大，越能把分離的文字和人黏在一起視為同一張圖。")
    st.sidebar.info("💡 此模式現在會自動偵測「非綠色」區域進行切割。")
    
elif "自由" in slice_mode:
    st.sidebar.markdown("### 🔪 手術刀切割 (偏移校正)")
    
    with st.sidebar.expander("↕️ 直向切割線 (Vertical)", expanded=True):
        off_v1 = st.slider("線 1", -100, 100, 0)
        off_v2 = st.slider("線 2", -100, 100, 0)
        off_v3 = st.slider("線 3", -100, 100, 0)

    with st.sidebar.expander("↔️ 橫向切割線 (Horizontal)", expanded=True):
        off_h1 = st.slider("線 A", -100, 100, 0)
        off_h2 = st.slider("線 B", -100, 100, 0)

# --- 函數定義區 ---

def get_grid_lines(w, h, ov1, ov2, ov3, oh1, oh2):
    base_v1 = w // 4
    base_v2 = w * 2 // 4
    base_v3 = w * 3 // 4
    base_h1 = h // 3
    base_h2 = h * 2 // 3
    
    v_lines = [0, base_v1 + ov1, base_v2 + ov2, base_v3 + ov3, w]
    h_lines = [0, base_h1 + oh1, base_h2 + oh2, h]
    return v_lines, h_lines

def draw_freeline_preview(img_pil, v_lines, h_lines):
    img = img_pil.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for x in v_lines[1:-1]:
        draw.line([(x, 0), (x, h)], fill="red", width=5)
    for y in h_lines[1:-1]:
        draw.line([(0, y), (w, y)], fill="red", width=5)
    return img

def remove_green_screen_hsv(img_pil, sensitivity=50, white_protect=30):
    img = np.array(img_pil.convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat_threshold = 140 - int(sensitivity * 0.9) 
    lower_green = np.array([35, sat_threshold, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    if white_protect > 0:
        protect_s_max = int(white_protect * 0.8) 
        protect_v_min = 255 - int(white_protect * 1.5) 
        lower_white = np.array([0, 0, protect_v_min])      
        upper_white = np.array([180, protect_s_max, 255]) 
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        final_green_mask = cv2.bitwise_and(green_mask, cv2.bitwise_not(white_mask))
    else:
        final_green_mask = green_mask

    img_rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
    img_rgba[final_green_mask > 0] = (0, 0, 0, 0)
    return Image.fromarray(cv2.cvtColor(img_rgba, cv2.COLOR_BGRA2RGBA))

def add_stroke(img_pil, thickness=8, color=(255, 255, 255, 255)):
    if thickness == 0: return img_pil
    img = img_pil.convert("RGBA")
    r, g, b, a = img.split()
    alpha_np = np.array(a)
    kernel = np.ones((thickness * 2 + 1, thickness * 2 + 1), np.uint8)
    outline_alpha = cv2.dilate(alpha_np, kernel, iterations=1)
    stroke_bg = Image.new("RGBA", img.size, color)
    stroke_bg.putalpha(Image.fromarray(outline_alpha))
    final_img = Image.alpha_composite(stroke_bg, img)
    return final_img

def extract_and_resize(sticker_img_pil, mode_selection, sensitivity, protect, border, edge_crop_px):
    if "綠幕" in mode_selection:
        sticker_no_bg = remove_green_screen_hsv(sticker_img_pil, sensitivity, protect)
    else:
        sticker_no_bg = remove(sticker_img_pil)
    
    if edge_crop_px > 0:
        w, h = sticker_no_bg.size
        if w > edge_crop_px*2 and h > edge_crop_px*2:
            sticker_no_bg = sticker_no_bg.crop((edge_crop_px, edge_crop_px, w - edge_crop_px, h - edge_crop_px))
    
    bbox = sticker_no_bg.getbbox()
    if bbox:
        sticker_cropped = sticker_no_bg.crop(bbox)
        if border > 0: sticker_cropped = add_stroke(sticker_cropped, border)
        sticker_cropped.thumbnail((370, 320), Image.Resampling.LANCZOS)
        final_bg = Image.new("RGBA", (370, 320), (0, 0, 0, 0))
        left = (370 - sticker_cropped.width) // 2
        top = (320 - sticker_cropped.height) // 2
        final_bg.paste(sticker_cropped, (left, top))
        return final_bg
    return None

def create_checkerboard_bg(size, grid_size=20):
    bg = Image.new("RGB", size, (220, 220, 220))
    draw = ImageDraw.Draw(bg)
    for y in range(0, size[1], grid_size):
        for x in range(0, size[0], grid_size):
            if (x // grid_size + y // grid_size) % 2 == 0:
                draw.rectangle([x, y, x+grid_size, y+grid_size], fill=(255, 255, 255))
    return bg

def process_single_image(image_pil, mode_selection, slicing_strategy, dilation_val=25, sensitivity=50, protect=30, border=8, edge_crop_px=0, 
                         ov1=0, ov2=0, ov3=0, oh1=0, oh2=0):
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    processed_stickers = []
    
    if "智慧" in slicing_strategy:
        # v10.0 更新：針對綠幕優化的智慧偵測
        if "綠幕" in mode_selection:
            # 1. 轉 HSV
            hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
            # 2. 定義綠色背景範圍 (與去背邏輯稍微不同，這裡要抓「背景」)
            lower_green = np.array([35, 40, 40])
            upper_green = np.array([85, 255, 255])
            bg_mask = cv2.inRange(hsv, lower_green, upper_green)
            # 3. 反轉遮罩：非綠色 = 前景 (貼圖)
            thresh = cv2.bitwise_not(bg_mask)
        else:
            # 白底模式
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

        # 4. 膨脹 (黏合分離的元件)
        kernel = np.ones((dilation_val, dilation_val), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        
        # 5. 找輪廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 2000 # 稍微調高，過濾雜訊
        
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        bounding_boxes = [cv2.boundingRect(c) for c in valid_contours]
        
        # 排序：由上到下，由左到右
        bounding_boxes.sort(key=lambda x: (round(x[1]/100), x[0]))
        
        for x, y, w, h in bounding_boxes:
            sticker_cv = img_cv[y:y+h, x:x+w]
            sticker_pil = Image.fromarray(cv2.cvtColor(sticker_cv, cv2.COLOR_BGR2RGB))
            result = extract_and_resize(sticker_pil, mode_selection, sensitivity, protect, border, edge_crop_px)
            if result: processed_stickers.append(result)
            
        return processed_stickers, ("Smart", "Vision")

    else:
        # 網格切割
        height, width, _ = img_cv.shape
        v_lines, h_lines = get_grid_lines(width, height, ov1, ov2, ov3, oh1, oh2)
        
        for r in range(3):
            for c in range(4):
                x_start = v_lines[c]
                x_end = v_lines[c+1]
                y_start = h_lines[r]
                y_end = h_lines[r+1]
                
                if x_end <= x_start or y_end <= y_start: continue
                sticker_cv = img_cv[y_start:y_end, x_start:x_end]
                if sticker_cv.size == 0: continue
                    
                sticker_pil = Image.fromarray(cv2.cvtColor(sticker_cv, cv2.COLOR_BGR2RGB))
                result = extract_and_resize(sticker_pil, mode_selection, sensitivity, protect, border, edge_crop_px)
                if result: processed_stickers.append(result)

        return processed_stickers, ("3", "4")

def create_resized_image(img, target_size):
    img = img.copy()
    img.thumbnail(target_size, Image.Resampling.LANCZOS)
    bg = Image.new("RGBA", target_size, (0, 0, 0, 0))
    left = (target_size[0] - img.width) // 2
    top = (target_size[1] - img.height) // 2
    bg.paste(img, (left, top))
    return bg

# --- 主程式區 ---

# 1. 預覽區 (自由切割模式才顯示)
if uploaded_files and "自由" in slice_mode:
    st.divider()
    st.header("👀 切割線即時預覽")
    st.info("請調整側邊欄的滑桿，確保**紅色線條**沒有切到角色或對話框。")
    first_img = Image.open(uploaded_files[0]).convert("RGB")
    w, h = first_img.size
    v_preview, h_preview = get_grid_lines(w, h, off_v1, off_v2, off_v3, off_h1, off_h2)
    preview_img = draw_freeline_preview(first_img, v_preview, h_preview)
    st.image(preview_img, caption="紅線切割預覽 (即時)", use_container_width=True)
    st.divider()

# 2. 執行區
if run_button:
    if not uploaded_files:
        st.error("❌ 請先上傳圖片！")
    else:
        st.toast("🚀 開始處理...", icon="🔥")
        st.session_state.processed_stickers = []
        st.session_state.original_images = []
        
        with st.status("正在全力加工中...", expanded=True) as status:
            progress_bar = st.progress(0)
            try:
                for idx, uploaded_file in enumerate(uploaded_files):
                    st.write(f"📥 讀取：{uploaded_file.name}")
                    image = Image.open(uploaded_file).convert("RGB")
                    st.session_state.original_images.append((uploaded_file.name, image))
                    
                    stickers, strategy_used = process_single_image(
                        image, remove_mode, slice_mode, dilation_size, 
                        sensitivity=gs_sensitivity, protect=highlight_protection, border=border_thickness, edge_crop_px=edge_crop,
                        ov1=off_v1, ov2=off_v2, ov3=off_v3, oh1=off_h1, oh2=off_h2
                    )
                    
                    st.write(f"✅ 完成 (產出 {len(stickers)} 張)")
                    st.session_state.processed_stickers.extend(stickers)
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                if not st.session_state.processed_stickers:
                    status.update(label="⚠️ 沒切出東西", state="error")
                    st.error("⚠️ 未偵測到貼圖，請檢查：1.去背模式是否選對 2.膨脹係數是否需要調整")
                else:
                    status.update(label="✅ 處理完成！", state="complete", expanded=False)
                    st.success(f"🎉 成功！共產出 {len(st.session_state.processed_stickers)} 張貼圖。")
                    
            except Exception as e:
                status.update(label="❌ 發生錯誤", state="error")
                st.error(f"程式執行錯誤: {e}")

# 3. 成果區
if st.session_state.processed_stickers:
    st.divider()
    st.header("🖼️ 貼圖總覽與設定")
    
    total_stickers = len(st.session_state.processed_stickers)
    sticker_options = [f"{i+1:02d}" for i in range(total_stickers)]
    
    col_selectors, col_preview = st.columns([1, 2])
    
    preview_bg_main = create_checkerboard_bg((240, 240), grid_size=20)
    preview_bg_tab = create_checkerboard_bg((96, 74), grid_size=10)

    with col_selectors:
        st.subheader("設定關鍵圖片")
        main_idx = int(st.selectbox("⭐ Main 圖片", sticker_options, index=0)) - 1
        tab_idx = int(st.selectbox("🏷️ Tab 圖片", sticker_options, index=0)) - 1
        
        main_img = create_resized_image(st.session_state.processed_stickers[main_idx], (240, 240))
        tab_img = create_resized_image(st.session_state.processed_stickers[tab_idx], (96, 74))
        
        disp_main = preview_bg_main.copy()
        disp_main.paste(main_img, (0,0), main_img)
        
        disp_tab = preview_bg_tab.copy()
        disp_tab.paste(tab_img, (0,0), tab_img)
        
        c1, c2 = st.columns(2)
        c1.image(disp_main, caption="Main (預覽)")
        c2.image(disp_tab, caption="Tab (預覽)")

    with col_preview:
        st.subheader("全部預覽 (棋盤底檢視)")
        preview_cols = st.columns(6)
        standard_bg = create_checkerboard_bg((370, 320), grid_size=32)

        for i, sticker in enumerate(st.session_state.processed_stickers):
            with preview_cols[i % 6]:
                disp_img = standard_bg.copy()
                disp_img.paste(sticker, (0, 0), sticker)
                st.image(disp_img, caption=f"{i+1:02d}", use_container_width=True)

    st.divider()
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for name, img in st.session_state.original_images:
            img_byte = io.BytesIO()
            img.save(img_byte, format='PNG')
            zf.writestr(f"Originals/{name.replace('.jpg','.png')}", img_byte.getvalue())
        for i, sticker in enumerate(st.session_state.processed_stickers):
            sticker_byte = io.BytesIO()
            sticker.save(sticker_byte, format='PNG')
            zf.writestr(f"Stickers/{i+1:02d}.png", sticker_byte.getvalue())
        
        main_byte = io.BytesIO()
        main_img.save(main_byte, format='PNG')
        zf.writestr("main.png", main_byte.getvalue())
        tab_byte = io.BytesIO()
        tab_img.save(tab_byte, format='PNG')
        zf.writestr("tab.png", tab_byte.getvalue())

    st.download_button(
        label=f"📦 下載 v10.0 完整懶人包",
        data=zip_buffer.getvalue(),
        file_name="SarahDad_v10.0_Stickers.zip",
        mime="application/zip",
        type="primary"
    )
