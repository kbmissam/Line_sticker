import streamlit as st
from PIL import Image, ImageDraw, ImageOps, ImageFilter
from rembg import remove
import io
import zipfile
import numpy as np
import cv2
import math

# --- 頁面設定 ---
st.set_page_config(page_title="莎拉爸貼圖神器 v15.0", page_icon="🍌", layout="wide")
st.title("🍌 莎拉爸貼圖神器 v15.0 (全能完全體)")
st.markdown("""
🚀 **v15.0 更新**：
1. **視覺膠水 (Visual Glue)**：防止文字被切斷。
2. **二次構圖 (Re-Composition)**：切圖後可 **無損放大** 與 **調整位置**，徹底解決角色太小的問題。
""")

# --- Session State ---
if 'processed_stickers' not in st.session_state: st.session_state.processed_stickers = []
if 'original_images' not in st.session_state: st.session_state.original_images = []
if 'uploader_key' not in st.session_state: st.session_state.uploader_key = 0

# --- 側邊欄 ---
st.sidebar.header("⚙️ 控制台")
if st.sidebar.button("🗑️ 清除重來", type="secondary", use_container_width=True):
    st.session_state.processed_stickers = []
    st.session_state.original_images = []
    st.session_state.uploader_key += 1 
    st.rerun()
run_button = st.sidebar.button("🚀 開始處理圖片", type="primary", use_container_width=True)
st.sidebar.markdown("---")

# 設定區
uploaded_files = st.sidebar.file_uploader("1. 上傳圖片", type=["jpg", "png"], accept_multiple_files=True, key=f"uploader_{st.session_state.uploader_key}")

st.sidebar.header("2. 影視級去背設定")
remove_mode = st.sidebar.radio("去背核心", ("🟢 綠幕模式 (Pro Despill)", "🤖 AI 模式 (rembg)"))

# 變數初始化
gs_sensitivity = 50
highlight_protection = 30
despill_level = 0.8
edge_softness = 3
mask_erode = 1

if "綠幕" in remove_mode:
    st.sidebar.markdown("##### 🎨 色彩與邊緣處理")
    gs_sensitivity = st.sidebar.slider("🟢 綠色閥值 (Sensitivity)", 0, 100, 50, help="數值越高，綠色去得越狠。")
    highlight_protection = st.sidebar.slider("💡 亮部保護", 0, 100, 30, help="防止誤刪白襯衫或眼白。")
    
    st.sidebar.markdown("##### 🧼 淨化工具")
    despill_level = st.sidebar.slider("🧪 去綠邊強度 (Despill)", 0.0, 1.0, 0.8, help="消除邊緣綠色光暈。")
    mask_erode = st.sidebar.slider("🤏 遮罩內縮 (Choke)", 0, 5, 1, help="物理消除綠邊。")
    edge_softness = st.sidebar.slider("☁️ 邊緣羽化 (Softness)", 0, 10, 3, help="消除鋸齒。")

st.sidebar.header("3. 裝飾與切割 (v14 核心)")
border_thickness = st.sidebar.slider("⚪ 白邊厚度", 0, 20, 8)

st.sidebar.markdown("##### 🧩 切割策略")
slice_mode = st.sidebar.radio("模式", ("🎯 智能網格 (Auto Grid 4x3)", "🧠 純智慧視覺", "📏 自由多線"))

grid_padding = 50
dilation_strength = 25 

if "智能網格" in slice_mode or "純智慧視覺" in slice_mode:
    grid_padding = st.sidebar.slider("↔️ 裁切寬容度 (Padding)", 10, 150, 50, help="切完後要留多少邊距。")
    
    # --- v14：視覺膠水控制 ---
    st.sidebar.markdown("##### 🧪 視覺膠水 (Visual Glue)")
    dilation_strength = st.sidebar.slider("🎈 膨脹係數 (Dilation)", 5, 100, 40, help="數值越大，膠水越強，能把離很遠的文字跟身體黏在一起切下來。")

# --- v15：二次構圖控制 ---
st.sidebar.markdown("---")
st.sidebar.header("4. 二次構圖 (v15 核心)")
st.sidebar.markdown("解決「角色太小」的問題，切圖後自動放大裁切。")
zoom_factor = st.sidebar.slider("🔎 放大倍率 (Zoom)", 1.0, 2.5, 1.0, 0.1, help="1.0=原圖。1.5=放大50%。拉大此數值可讓角色特寫更明顯。")
offset_y = st.sidebar.slider("↕️ 垂直位移 (Offset Y)", -100, 100, 0, step=5, help="正數向下移，負數向上移。如果放大後頭頂被切到，請往下拉(正數)。")


# --- 核心演算法區 ---

def apply_despill(img_bgr, strength=0.8):
    """專業級 Despill 演算法"""
    img_float = img_bgr.astype(np.float32)
    b, g, r = cv2.split(img_float)
    rb_avg = (r + b) / 2.0
    spill_mask = g > rb_avg
    g[spill_mask] = g[spill_mask] * (1 - strength) + rb_avg[spill_mask] * strength
    despilled = cv2.merge([b, g, r])
    return np.clip(despilled, 0, 255).astype(np.uint8)

def get_pro_matte(chunk_cv, sensitivity, protect, erode_iter, softness):
    """生成高品質 Alpha 遮罩"""
    hsv = cv2.cvtColor(chunk_cv, cv2.COLOR_BGR2HSV)
    sat_threshold = 140 - int(sensitivity * 0.9)
    lower_green = np.array([35, sat_threshold, 40])
    upper_green = np.array([85, 255, 255])
    bg_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    if protect > 0:
        s_max = int(protect * 0.8)
        v_min = 255 - int(protect * 1.5)
        lower_white = np.array([0, 0, v_min])
        upper_white = np.array([180, s_max, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        bg_mask = cv2.bitwise_and(bg_mask, cv2.bitwise_not(white_mask))
    
    fg_mask = cv2.bitwise_not(bg_mask)
    
    if erode_iter > 0:
        kernel = np.ones((3,3), np.uint8)
        fg_mask = cv2.erode(fg_mask, kernel, iterations=erode_iter)
        
    if softness > 0:
        k_size = softness * 2 + 1
        fg_mask = cv2.GaussianBlur(fg_mask, (k_size, k_size), 0)
        
    return fg_mask

def extract_content_smart_v14(chunk_cv, sensitivity, protect, d_strength, erode, soft, dilation_val):
    """v14 視覺膠水演算法"""
    h, w, _ = chunk_cv.shape
    
    # 1. 取得基礎遮罩
    base_mask = get_pro_matte(chunk_cv, sensitivity, protect, 0, 0)
    
    # --- v14 關鍵步驟：視覺膨脹 (The Glue) ---
    glue_kernel_size = dilation_val
    glue_kernel = np.ones((glue_kernel_size, glue_kernel_size), np.uint8)
    
    # 製作「偵測用遮罩」
    detection_mask = cv2.dilate(base_mask, glue_kernel, iterations=1)
    
    # 2. 在「膨脹遮罩」上找輪廓
    contours, _ = cv2.findContours(detection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours: return None
    
    # 過濾太小的雜訊
    min_area = 1000 
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    if not valid_contours: return None
    
    # 3. 找出「最大的那一坨」
    best_cnt = max(valid_contours, key=cv2.contourArea)
    
    # 4. 取得這一坨的座標 (Bounding Box)
    x, y, cw, ch = cv2.boundingRect(best_cnt)
    
    # 5. 製作最終的高畫質遮罩
    high_quality_mask = get_pro_matte(chunk_cv, sensitivity, protect, erode, soft)
    
    # 6. 應用 Despill
    if d_strength > 0:
        chunk_clean = apply_despill(chunk_cv, d_strength)
    else:
        chunk_clean = chunk_cv
        
    # 7. 合併 RGBA
    b, g, r = cv2.split(chunk_clean)
    rgba = cv2.merge([r, g, b, high_quality_mask])
    
    # 8. 裁切
    pad = soft + 2
    x_cut = max(0, x - pad)
    y_cut = max(0, y - pad)
    w_cut = min(w - x_cut, cw + x - x_cut + pad)
    h_cut = min(h - y_cut, ch + y - y_cut + pad)
    
    final_chunk = rgba[y_cut:y_cut+h_cut, x_cut:x_cut+w_cut]
    
    if final_chunk.size == 0: return None
    
    return Image.fromarray(final_chunk)

def add_stroke_and_resize(sticker_pil, border, zoom=1.0, offset_y=0):
    """
    v15.0 升級：加入 Zoom (放大) 與 Offset (位移) 功能
    """
    # 1. 先加白邊
    if border > 0:
        img = sticker_pil.convert("RGBA")
        r, g, b, a = img.split()
        alpha_np = np.array(a)
        
        kernel = np.ones((border * 2 + 1, border * 2 + 1), np.uint8)
        outline_alpha = cv2.dilate(alpha_np, kernel, iterations=1)
        
        stroke_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        stroke_bg.putalpha(Image.fromarray(outline_alpha))
        sticker_pil = Image.alpha_composite(stroke_bg, img)

    # 2. 二次構圖：放大裁切 (v15 New!)
    if zoom > 1.0 or offset_y != 0:
        # 取得目前尺寸
        w, h = sticker_pil.size
        
        # 計算放大後的尺寸
        new_w = int(w * zoom)
        new_h = int(h * zoom)
        
        # 進行高品質放大
        img_zoomed = sticker_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # 計算裁切範圍 (以中心為基準)
        # 目標是切回原始 w, h 的大小，但是中心點可以偏移
        left = (new_w - w) // 2
        top = (new_h - h) // 2
        
        # 應用垂直位移
        top = top - offset_y
        
        # 確保裁切框不超出範圍 (防呆)
        left = max(0, left)
        top = max(0, top)
        # 如果因為偏移導致下方超出範圍，也不要報錯，PIL crop 會自動處理邊界，
        # 但為了保險，我們設定 crop box
        right = left + w
        bottom = top + h
        
        sticker_pil = img_zoomed.crop((left, top, right, bottom))

    # 3. 縮放與置中 (LINE 貼圖標準 370x320)
    sticker_pil.thumbnail((370, 320), Image.Resampling.LANCZOS)
    final_bg = Image.new("RGBA", (370, 320), (0, 0, 0, 0))
    left = (370 - sticker_pil.width) // 2
    top = (320 - sticker_pil.height) // 2
    final_bg.paste(sticker_pil, (left, top))
    return final_bg

def process_image(image_pil, slice_strategy, padding, sens, prot, border, d_str, ero, soft, dilation_val, zoom, off_y):
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    # Pre-crop
    h_full, w_full, _ = img_cv.shape
    img_cv = img_cv[10:h_full-10, 10:w_full-10]
    
    results = []
    
    # --- 策略 A: 智能網格 (4x3) ---
    if "智能網格" in slice_strategy:
        h, w, _ = img_cv.shape
        v_lines = [int(w * i / 4) for i in range(5)]
        h_lines = [int(h * i / 3) for i in range(4)]
        
        for r in range(3):
            for c in range(4):
                x1, x2 = v_lines[c], v_lines[c+1]
                y1, y2 = h_lines[r], h_lines[r+1]
                
                x1_p = max(0, x1 - padding)
                x2_p = min(w, x2 + padding)
                y1_p = max(0, y1 - padding)
                y2_p = min(h, y2 + padding)
                
                chunk = img_cv[y1_p:y2_p, x1_p:x2_p]
                
                # 呼叫 v14 膠水函數
                sticker = extract_content_smart_v14(chunk, sens, prot, d_str, ero, soft, dilation_val)
                
                if sticker:
                    # 呼叫 v15 二次構圖函數 (傳入 zoom 與 offset)
                    final = add_stroke_and_resize(sticker, border, zoom, off_y)
                    results.append(final)
    
    # --- 策略 B: 純智慧視覺 (保留給不規則排列) ---
    elif "純智慧視覺" in slice_strategy:
         # 暫不支援網格外的邏輯
         pass 

    return results

def create_resized_image(img, target_size):
    img = img.copy()
    img.thumbnail(target_size, Image.Resampling.LANCZOS)
    bg = Image.new("RGBA", target_size, (0, 0, 0, 0))
    left = (target_size[0] - img.width) // 2
    top = (target_size[1] - img.height) // 2
    bg.paste(img, (left, top))
    return bg
    
def create_checkerboard_bg(size, grid_size=20):
    bg = Image.new("RGB", size, (220, 220, 220))
    draw = ImageDraw.Draw(bg)
    for y in range(0, size[1], grid_size):
        for x in range(0, size[0], grid_size):
            if (x // grid_size + y // grid_size) % 2 == 0:
                draw.rectangle([x, y, x+grid_size, y+grid_size], fill=(255, 255, 255))
    return bg

# --- 主程式 ---
if run_button:
    if not uploaded_files:
        st.error("❌ 請先上傳圖片！")
    else:
        st.toast("🚀 啟動 v15 全能引擎...", icon="✨")
        st.session_state.processed_stickers = []
        st.session_state.original_images = []
        
        with st.status("正在進行 AI 切割、文字黏合與二次構圖...", expanded=True) as status:
            prog = st.progress(0)
            for i, f in enumerate(uploaded_files):
                img = Image.open(f).convert("RGB")
                st.session_state.original_images.append((f.name, img))
                
                res = process_image(
                    img, slice_mode, grid_padding, 
                    gs_sensitivity, highlight_protection, border_thickness,
                    despill_level, mask_erode, edge_softness, dilation_strength,
                    zoom_factor, offset_y # v15 新參數
                )
                st.session_state.processed_stickers.extend(res)
                prog.progress((i+1)/len(uploaded_files))
            
            if st.session_state.processed_stickers:
                status.update(label="✅ 處理完成", state="complete", expanded=False)
                st.success(f"🎉 成功產出 {len(st.session_state.processed_stickers)} 張貼圖！(已套用放大與位移)")
            else:
                status.update(label="⚠️ 失敗", state="error")
                st.error("未偵測到貼圖，請嘗試調大「綠色閥值」或「膨脹係數」。")

# 預覽區
if st.session_state.processed_stickers:
    st.divider()
    st.header("🖼️ 成果預覽")
    
    opts = [f"{i+1:02d}" for i in range(len(st.session_state.processed_stickers))]
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.subheader("設定縮圖")
        if opts:
            m_idx = int(st.selectbox("Main", opts, index=0)) - 1
            t_idx = int(st.selectbox("Tab", opts, index=0)) - 1
            
            m_img = create_resized_image(st.session_state.processed_stickers[m_idx], (240, 240))
            t_img = create_resized_image(st.session_state.processed_stickers[t_idx], (96, 74))
            
            bg_m = create_checkerboard_bg((240, 240))
            bg_m.paste(m_img, (0,0), m_img)
            st.image(bg_m, caption="Main")
            
            bg_t = create_checkerboard_bg((96, 74), 10)
            bg_t.paste(t_img, (0,0), t_img)
            st.image(bg_t, caption="Tab")

    with c2:
        st.subheader("全部貼圖")
        cols = st.columns(6)
        bg = create_checkerboard_bg((370, 320), 32)
        for i, s in enumerate(st.session_state.processed_stickers):
            disp = bg.copy()
            disp.paste(s, (0,0), s)
            cols[i%6].image(disp, caption=f"{i+1:02d}", use_container_width=True)

    st.divider()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for i, s in enumerate(st.session_state.processed_stickers):
            b = io.BytesIO()
            s.save(b, "PNG")
            zf.writestr(f"Stickers/{i+1:02d}.png", b.getvalue())
        
        if opts:
            bm, bt = io.BytesIO(), io.BytesIO()
            m_img.save(bm, "PNG"); zf.writestr("main.png", bm.getvalue())
            t_img.save(bt, "PNG"); zf.writestr("tab.png", bt.getvalue())
            
    st.download_button("📦 下載 v15.0 懶人包", buf.getvalue(), "SarahDad_v15.0.zip", "application/zip", type="primary")
