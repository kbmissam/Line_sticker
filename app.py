import streamlit as st
from PIL import Image, ImageDraw, ImageOps, ImageFilter
from rembg import remove
import io
import zipfile
import numpy as np
import cv2

# --- 頁面設定 ---
st.set_page_config(page_title="莎拉爸貼圖神器 v7.7", page_icon="🐴", layout="wide")
st.title("🐴 莎拉爸貼圖神器 v7.7 (完整修復版)")
st.markdown("🚀 **v7.7 更新**：依據 Claude 健檢報告修復函數截斷與變數邏輯，針對 4x3 批次優化預設值。")

# --- Session State 初始化 ---
if 'processed_stickers' not in st.session_state:
    st.session_state.processed_stickers = []
if 'original_images' not in st.session_state:
    st.session_state.original_images = []
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

# --- 側邊欄：控制台 ---
st.sidebar.header("⚙️ 控制台")

# 清除按鈕邏輯
if st.sidebar.button("🗑️ 清除重來 (Reset All)", type="secondary", use_container_width=True):
    st.session_state.processed_stickers = []
    st.session_state.original_images = []
    st.session_state.uploader_key += 1 
    st.rerun()

# 執行按鈕
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

# 初始化變數 (避免 Claude 指出的作用域問題)
gs_sensitivity = 50
highlight_protection = 30 
border_thickness = 8

if "綠幕" in remove_mode:
    st.sidebar.markdown("##### 🔧 去背微調 (修復破洞)")
    gs_sensitivity = st.sidebar.slider(
        "🟢 綠幕敏感度", 0, 100, 50, 
        help="數值越小越安全，數值越大去得越乾淨但可能破洞。"
    )
    highlight_protection = st.sidebar.slider(
        "💡 亮部保護", 0, 100, 30, 
        help="保護白色反光不被切掉。數值越高保護越強。"
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

# 針對您現在的 4x3 需求，將預設值調整為 3, 4
manual_rows, manual_cols = 3, 4
dilation_size = 25

if "智慧" in slice_mode:
    dilation_size = st.sidebar.slider("膨脹係數 (防切字)", 5, 50, 25)
    st.sidebar.info("💡 適合：排列不規則，但間距必須足夠。")
elif "自動" in slice_mode:
    st.sidebar.success("✨ 程式將根據圖片長寬比，自動決定是用 6x5 還是 8x5 切割。")
else:
    st.sidebar.warning("⚠️ 手動模式預設為 3x4 (適合小籠包批次)。")
    c1, c2 = st.sidebar.columns(2)
    with c1:
        manual_rows = st.number_input("縱向列數 (Rows)", 1, 10, 3) 
    with c2:
        manual_cols = st.number_input("橫向行數 (Cols)", 1, 10, 4) 

# --- 核心函數定義區 (完整無截斷) ---

def remove_green_screen_hsv(img_pil, sensitivity=50, white_protect=30):
    # 這是 Claude 指出之前被截斷的函數，現在完整補上
    img = np.array(img_pil.convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # 綠色範圍定義
    sat_threshold = 140 - int(sensitivity * 0.9) 
    lower_green = np.array([35, sat_threshold, 40])
    upper_green = np.array([85, 255, 255])
    
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # 亮部保護邏輯
    if white_protect > 0:
        protect_s_max = int(white_protect * 0.8) 
        protect_v_min = 255 - int(white_protect * 1.5) 
        lower_white = np.array([0, 0, protect_v_min])      
        upper_white = np.array([180, protect_s_max, 255]) 
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        # 從綠色遮罩中扣除白色保護區
        final_green_mask = cv2.bitwise_and(green_mask, cv2.bitwise_not(white_mask))
    else:
        final_green_mask = green_mask

    # 轉為透明背景
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

def extract_and_resize(sticker_img_pil, mode_selection, sensitivity, protect, border):
    # 1. 去背
    if "綠幕" in mode_selection:
        sticker_no_bg = remove_green_screen_hsv(sticker_img_pil, sensitivity, protect)
    else:
        sticker_no_bg = remove(sticker_img_pil)
    
    # 2. 裁切與後製
    bbox = sticker_no_bg.getbbox()
    if bbox:
        sticker_cropped = sticker_no_bg.crop(bbox)
        if border > 0: sticker_cropped = add_stroke(sticker_cropped, border)
        
        # 3. 縮放與畫布補正 (強制 370x320 偶數)
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

def process_single_image(image_pil, mode_selection, slicing_strategy, dilation_val=25, man_r=3, man_c=4, sensitivity=50, protect=30, border=8):
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    processed_stickers = []
    
    use_grid = False
    grid_rows, grid_cols = man_r, man_c # 預設使用傳入的手動參數
    
    if "智慧" in slicing_strategy: use_grid = False
    elif "手動" in slicing_strategy: use_grid = True
    elif "自動" in slicing_strategy:
        use_grid = True
        h, w, _ = img_cv.shape
        ratio = w / h
        if ratio > 1.4: grid_rows, grid_cols = 5, 8
        else: grid_rows, grid_cols = 5, 6

    # 切割邏輯
    if not use_grid:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
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
            result = extract_and_resize(sticker_pil, mode_selection, sensitivity, protect, border)
            if result: processed_stickers.append(result)
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

# --- 主程式執行區 ---

# 確保按鈕被按下時有明顯反應
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
                        manual_rows, manual_cols,
                        gs_sensitivity, highlight_protection, border_thickness
                    )
                    
                    st.write(f"✅ 完成 (模式: {strategy_used[1]}x{strategy_used[0]}, 產出 {len(stickers)} 張)")
                    st.session_state.processed_stickers.extend(stickers)
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                if not st.session_state.processed_stickers:
                    status.update(label="⚠️ 沒切出東西，請檢查設定", state="error")
                    st.error("⚠️ 未偵測到貼圖，請檢查切割策略或去背設定。")
                else:
                    status.update(label="✅ 處理完成！", state="complete", expanded=False)
                    st.success(f"🎉 成功！共產出 {len(st.session_state.processed_stickers)} 張貼圖。")
                    
            except Exception as e:
                status.update(label="❌ 發生錯誤", state="error")
                st.error(f"程式執行錯誤: {e}")

# --- 預覽與下載區 ---

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
        label=f"📦 下載 v7.7 完整懶人包",
        data=zip_buffer.getvalue(),
        file_name="SarahDad_v7.7_Stickers.zip",
        mime="application/zip",
        type="primary"
    )
