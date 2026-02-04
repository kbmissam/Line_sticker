import streamlit as st
from PIL import Image
from rembg import remove
import io
import zipfile
import numpy as np
import cv2

# --- 頁面設定 ---
st.set_page_config(page_title="莎拉爸貼圖神器 v7.0", page_icon="🐴", layout="wide")
st.title("🐴 莎拉爸貼圖神器 v7.0 (最終穩定版)")
st.markdown("🚀 **v7.0 更新**：新增「綠幕敏感度」滑桿與 HSV 專業演算法，徹底解決去背問題！")

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

st.sidebar.header("2. 去背與效果")
remove_mode = st.sidebar.radio(
    "選擇去背方式：",
    ("🟢 綠幕模式 (可調靈敏度)", "🤖 AI 模式 (白底用)")
)

# [v7.0 新增] 綠幕調整滑桿
gs_sensitivity = 0
if "綠幕" in remove_mode:
    st.sidebar.markdown("##### 🔧 綠幕設定")
    gs_sensitivity = st.sidebar.slider(
        "綠幕敏感度 (Sensitivity)", 
        min_value=10, max_value=100, value=60, 
        help="數值越高越嚴格 (去更多綠色)；數值越低越寬容 (保留更多細節)。若角色破洞請調低，若背景去不掉請調高。"
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

# [v7.0 升級] HSV 專業去背算法
def remove_green_screen_hsv(img_pil, sensitivity=60):
    # 轉換為 OpenCV BGR 格式
    img = np.array(img_pil.convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # 轉換為 HSV 格式 (色相、飽和度、亮度)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # 定義綠色的 HSV 範圍
    # H: 綠色大約在 60 度左右。OpenCV 中 H 範圍是 0-180。
    # 敏感度控制範圍寬度。
    
    # 基礎綠色範圍 (這通常是螢光綠的落點)
    lower_green = np.array([35, 100, 100]) 
    upper_green = np.array([85, 255, 255])
    
    # 根據敏感度微調 (Slider 越高，範圍越寬，抓得越嚴格)
    # 這裡我們用 Slider 來調整 "飽和度(S)" 和 "亮度(V)" 的下限
    # 敏感度低 (10) -> S, V 下限高 -> 必須是非常亮、非常鮮豔的綠才去除 (寬容)
    # 敏感度高 (100) -> S, V 下限低 -> 暗綠色也會被去除 (嚴格)
    
    s_floor = 255 - int(sensitivity * 2.5) # 10->230 (Strict), 100->5 (Loose)? No wait.
    # 重新設計邏輯：
    # Sensitivity 高 = 認定更多東西是背景 = Mask 範圍大
    # Sensitivity 低 = 認定更少東西是背景 = Mask 範圍小
    
    # 螢光綠是 (60, 255, 255)
    # 寬容度變數
    tolerance = int(sensitivity * 0.8) # 10->8, 60->48, 100->80
    
    lower_green = np.array([60 - 30, 40 + tolerance, 40 + tolerance])
    upper_green = np.array([60 + 30, 255, 255])
    
    # 建立遮罩
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # 將遮罩反轉 (黑色是背景，白色是保留)
    mask_inv = cv2.bitwise_not(mask)
    
    # 轉回 RGBA
    img_rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGBA)
    # 將遮罩應用到 Alpha 通道
    img_rgba[:, :, 3] = mask_inv
    
    return Image.fromarray(img_rgba)


def add_white_border(image_pil, thickness):
    """為透明背景的圖片加上白色描邊"""
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
        return image_pil # 如果出錯，回傳原圖

def process_single_image(image_pil, mode_selection, slicing_strategy, dilation_val, man_r, man_c, border_thick, gs_sens):
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
            # 使用新的 HSV 算法 + 敏感度參數
            sticker_no_bg = remove_green_screen_hsv(sticker_pil_raw, gs_sens)
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
            if w_new == 0 or h_new == 0: return None # 防止空圖
            
            if w_new != sticker_final.width or h_new != sticker_final.height:
                 sticker_final = sticker_final.resize((w_new, h_new), Image.Resampling.LANCZOS)
            return sticker_final
        return None

    # --- 執行切割 ---
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

def create_resized_image(img, target_size):
    try:
        img = img.copy()
        img.thumbnail(target_size, Image.Resampling.LANCZOS)
        bg = Image.new("RGBA", target_size, (0, 0, 0, 0))
        left = (target_size[0] - img.width) // 2
        top = (target_size[1] - img.height) // 2
        bg.paste(img, (left, top))
        return bg
    except:
        return Image.new("RGBA", target_size, (0,0,0,0)) # 出錯回傳空圖

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
                
                # 呼叫處理函數，傳入所有參數
                stickers, strategy_used = process_single_image(
                    image, remove_mode, slice_mode, dilation_size, 
                    manual_rows, manual_cols, border_thickness, gs_sensitivity
                )
                
                status_text.text(f"正在處理：{uploaded_file.name} ...")
                st.session_state.processed_stickers.extend(stickers)
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            if not st.session_state.processed_stickers:
                st.error("⚠️ 未偵測到貼圖。請檢查「綠幕敏感度」是否太高（導致全被切掉），或切割網格是否正確。")
            else:
                st.success(f"✅ 完成！共 {len(st.session_state.processed_stickers)} 張。")
                
        except Exception as e:
            st.error(f"處理過程發生錯誤: {e}")

# --- 預覽與下載區 ---
if st.session_state.processed_stickers:
    st.divider()
    st.header("🖼️ 貼圖總覽")
    
    try:
        total_stickers = len(st.session_state.processed_stickers)
        sticker_options = [f"{i+1:02d}" for i in range(total_stickers)]
        
        col_selectors, col_preview = st.columns([1, 2])
        
        with col_selectors:
            st.subheader("設定 Main/Tab")
            # 增加安全判斷，防止空列表導致 crash
            if sticker_options:
                main_idx = int(st.selectbox("⭐ Main 圖片", sticker_options, index=0)) - 1
                tab_idx = int(st.selectbox("🏷️ Tab 圖片", sticker_options, index=0)) - 1
                
                main_img = create_resized_image(st.session_state.processed_stickers[main_idx], (240, 240))
                tab_img = create_resized_image(st.session_state.processed_stickers[tab_idx], (96, 74))
                
                c1, c2 = st.columns(2)
                c1.image(main_img, caption="Main")
                c2.image(tab_img, caption="Tab")
            else:
                st.warning("沒有可用的貼圖選項。")

        with col_preview:
            st.subheader("預覽牆")
            preview_cols = st.columns(6)
            for i, sticker in enumerate(st.session_state.processed_stickers):
                with preview_cols[i % 6]:
                    st.image(sticker, caption=f"{i+1:02d}", use_container_width=True)
        
        st.divider()
        # 下載按鈕邏輯
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
            
            if 'main_img' in locals():
                main_byte = io.BytesIO()
                main_img.save(main_byte, format='PNG')
                zf.writestr("main.png", main_byte.getvalue())
            if 'tab_img' in locals():
                tab_byte = io.BytesIO()
                tab_img.save(tab_byte, format='PNG')
                zf.writestr("tab.png", tab_byte.getvalue())

        st.download_button(
            label=f"📦 下載 ZIP (v7.0)",
            data=zip_buffer.getvalue(),
            file_name="SarahDad_Stickers_v7.0.zip",
            mime="application/zip",
            type="primary"
        )
    except Exception as e:
        st.error(f"顯示預覽時發生錯誤: {e}")
