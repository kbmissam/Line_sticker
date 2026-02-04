import streamlit as st
from PIL import Image
from rembg import remove
import io
import zipfile
import numpy as np
import cv2

# --- 頁面設定 ---
st.set_page_config(page_title="莎拉爸貼圖神器 v6.6", page_icon="🐴", layout="wide")
st.title("🐴 莎拉爸貼圖神器 v6.6 (一鍵重置版)")
st.markdown("🚀 **v6.6 更新**：新增「清除重來」按鈕，自動清空上傳區與暫存檔，方便連續作業！")

# --- Session State 初始化 ---
if 'processed_stickers' not in st.session_state:
    st.session_state.processed_stickers = []
if 'original_images' not in st.session_state:
    st.session_state.original_images = []
# [v6.6 新增] 用來控制上傳元件的 ID，改變它就能強制清空上傳區
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

# --- 側邊欄：功能區 ---
st.sidebar.header("⚙️ 控制台")

# [v6.6 新增] 清除按鈕 (放在最顯眼的地方)
if st.sidebar.button("🗑️ 清除重來 (Reset All)", type="primary"):
    st.session_state.processed_stickers = []
    st.session_state.original_images = []
    st.session_state.uploader_key += 1 # 關鍵：改變 Key 強制重繪上傳元件
    st.rerun() # 重新執行畫面

st.sidebar.markdown("---")
st.sidebar.header("1. 上傳圖片")

# [v6.6 修改] 加入 key 參數
uploaded_files = st.sidebar.file_uploader(
    "請上傳貼圖大圖 (可多選混搭)", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.uploader_key}" # 綁定動態 Key
)

st.sidebar.header("2. 去背模式")
remove_mode = st.sidebar.radio(
    "選擇去背方式：",
    ("🟢 綠幕模式 (推薦！)", "🤖 AI 模式 (白底用)")
)

st.sidebar.header("3. 切割策略")
slice_mode = st.sidebar.radio(
    "選擇切割方式：",
    (
        "🧠 智慧視覺偵測 (不限格數)", 
        "🤖 強制網格 (自動判斷 6x5 / 8x5)", 
        "📏 強制網格 (手動設定 4x3 等)"
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
    st.sidebar.warning("⚠️ 手動模式：請自行設定行列數。")
    c1, c2 = st.sidebar.columns(2)
    with c1:
        # [v6.5 延續] 預設為 3 (配合 4x3)
        manual_rows = st.number_input("縱向列數 (Rows)", 1, 10, 3) 
    with c2:
        # [v6.5 延續] 預設為 4 (配合 4x3)
        manual_cols = st.number_input("橫向行數 (Cols)", 1, 10, 4) 

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
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        if "綠幕" in mode_selection:
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((dilation_val, dilation_val), np.uint8)
        thresh
