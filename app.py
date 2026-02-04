import streamlit as st
from PIL import Image, ImageDraw
from rembg import remove
import io
import zipfile
import numpy as np
import cv2

# --- 頁面設定 ---
st.set_page_config(page_title="莎拉爸貼圖神器 v7.1", page_icon="🐴", layout="wide")
st.title("🐴 莎拉爸貼圖神器 v7.1 (透明檢視優化版)")
st.markdown("🚀 **v7.1 更新**：預覽區新增「棋盤格灰底」，讓您能看清白邊與透明度！(下載檔仍為透明)")

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
    ("🟢 綠幕模式 (可調靈敏度)", "🤖 AI 模式 (白底用)")
)

gs_sensitivity = 60
if "綠幕" in remove_mode:
    st.sidebar.markdown("##### 🔧 綠幕設定")
    gs_sensitivity = st.sidebar.slider(
        "綠幕敏感度 (Sensitivity)", 
        min_value=10, max_value=100, value=60, 
        help="數值越高越嚴格 (去更多綠色)；數值越低越寬容 (保留更多細節)。"
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

# [v7.0 核心] HSV 去背
def remove_green_screen_hsv(img_pil, sensitivity=60):
    img = np.array(img_pil.convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # 動態調整範圍
    tolerance = int(sensitivity * 0.8) 
    lower_green = np.array([60 - 30, 40 + tolerance, 40 + tolerance])
    upper_green = np.array([60 + 30, 255, 255])
    
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)
    
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
        final_img = Image.alpha_composite(
