import streamlit as st
from PIL import Image, ImageDraw, ImageOps, ImageFilter
from rembg import remove
import io
import zipfile
import numpy as np
import cv2

# --- 頁面設定 ---
st.set_page_config(page_title="莎拉爸貼圖神器 v9.0", page_icon="🐴", layout="wide")
st.title("🐴 莎拉爸貼圖神器 v9.0 (自由手術刀版)")
st.markdown("🚀 **v9.0 更新**：解鎖「多線獨立控制」，可單獨調整 3 條垂直線與 2 條水平線，完美避開長寬不一的角色。")

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
        "📏 自由多線微調 (推薦 Batch 4)", 
        "🧠 智慧視覺偵測", 
        "🤖 強制網格 (平均分配)"
    )
)

# --- v9.0 核心：多線獨立控制變數 ---
# 預設偏移量都為 0
off_v1, off_v2, off_v3 = 0, 0, 0
off_h1, off_h2 = 0, 0
dilation_size = 35

if "智慧" in slice_mode:
    dilation_size = st.sidebar.slider("膨脹係數 (防切散)", 5, 100, 35)
    st.sidebar.info("💡 智慧模式會自動偵測物體邊緣。")
    
elif "自由" in slice_mode:
    st.sidebar.markdown("### 🔪 手術刀切割 (偏移校正)")
    st.sidebar.info("請看著右側預覽圖，調整下方的線條位置。")
    
    with st.sidebar.expander("↕️ 直向切割線 (Vertical)", expanded=True):
        st.caption("調整垂直線 (左右移動)")
        off_v1 = st.slider("線 1 (第1-2欄之間)", -100, 100, 0)
        off_v2 = st.slider("線 2 (第2-3欄之間)", -100, 100, 0)
        off_v3 = st.slider("線 3 (第3-4欄之間)", -100, 100, 0)

    with st.sidebar.expander("↔️ 橫向切割線 (Horizontal)", expanded=True):
        st.caption("調整水平線 (上下移動)")
        off_h1 = st.slider("線 A (第1-2列之間)", -100, 100, 0)
        off_h2 = st.slider("線 B (第2-3列之間)", -100, 100, 0)

else:
    st.sidebar.warning("⚠️ 標準模式：平均切割 4x3 網格。")

# --- 核心函數 ---

def get_grid_lines(w, h, ov1, ov2, ov3, oh1, oh2):
    """計算所有切割線的絕對坐標"""
    # 預設平均值
    base_v1 = w // 4
    base_v2 = w * 2 // 4
    base_v3 = w * 3 // 4
    
    base_h1 = h // 3
    base_h2 = h * 2 // 3
    
    # 加上使用者偏移量
    v_lines = [0, base_v1 + ov1, base_v2 + ov2, base_v3 + ov3, w]
    h_lines = [0, base_h1 + oh1, base_h2 + oh2, h]
    
    return v_lines, h_lines

def draw_freeline_preview(img_pil, v_lines, h_lines):
    """繪製 v9.0 的自由線預覽"""
    img = img_pil.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    
    # 畫垂直線 (跳過頭尾 0 和 w)
    for x in v_lines[1:-1]:
        draw.line([(x, 0), (x, h)], fill="red", width=5)
    
    # 畫水平線 (跳過頭尾 0 和 h)
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

    img_rgba = cv2.cvtColor
