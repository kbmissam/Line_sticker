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
