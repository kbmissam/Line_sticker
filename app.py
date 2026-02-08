import streamlit as st
from PIL import Image, ImageDraw, ImageOps, ImageFilter
from rembg import remove
import io
import zipfile
import numpy as np
import cv2
import math

# --- 頁面設定 ---
st.set_page_config(page_title="莎拉爸貼圖神器 v15.1", page_icon="🍌", layout="wide")
st.title("🍌 莎拉爸貼圖神器 v15.1 (緊密裁切修正版)")
st.markdown("""
🚀 **v15.1 重要修正**：
修正了二次構圖的邏輯。現在會先**自動切除周圍多餘的透明區域**，抓緊角色與白邊，然後再進行放大與置中，確保角色真正撐滿貼圖空間。
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
    grid_padding = st.sidebar.slider("↔️ 裁切寬容度 (Padding)", 10, 150, 50, help="建議調小 (如 10-20)，配合下方自動緊密裁切效果更好。")
    
    # --- v14：視覺膠水控制 ---
    st.sidebar.markdown("##### 🧪 視覺膠水 (Visual Glue)")
    dilation_strength = st.sidebar.slider("🎈 膨脹係數 (Dilation)", 5, 100, 40, help="防止文字被切斷的膠水強度。")

# --- v15.1：二次構圖控制 (修正版) ---
st.sidebar.markdown("---")
st.sidebar.header("4. 二次構圖 (v15.1 修正)")
st.sidebar.markdown("現在會自動切除多餘透明邊框！")
zoom_factor = st.sidebar.slider("🔎 額外放大倍率 (Zoom)", 1.0, 2.0, 1.0, 0.1, help="在緊密裁切的基礎上，再額外放大角色。")
offset_y = st.sidebar.slider("↕️ 垂直位移 (Offset Y)", -100, 100, 0, step=5, help="正數向下移，負數向上移。調整角色在最終畫布中的位置。")


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
    base_mask = get_pro_matte(chunk_cv, sensitivity, protect, 0, 0)
    glue_kernel_size = dilation_val
    glue_kernel = np.ones((glue_kernel_size, glue_kernel_size), np.uint8)
    detection_mask = cv2.dilate(base_mask, glue_kernel, iterations=1)
    contours, _ = cv2.findContours(detection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    min_area = 1000 
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    if not valid_contours: return None
    best_cnt = max(valid_contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(best_cnt)
    high_quality_mask = get_pro_matte(chunk_cv, sensitivity, protect, erode, soft)
    if d_strength > 0:
        chunk_clean = apply_despill(chunk_cv, d_strength)
    else:
        chunk_clean = chunk_cv
    b, g, r = cv2.split(chunk_clean)
    rgba = cv2.merge([r, g, b, high_quality_mask])
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
    v15.1 核心修正：自動緊密裁切 + 放大置中
    解決角色太小以及透明邊框過多的問題。
    """
    # 1. 先加白邊 (維持原邏輯)
    img_rgba = sticker_pil.convert("RGBA")
    if border > 0:
        r, g, b, a = img_rgba.split()
        alpha_np = np.array(a)
        kernel = np.ones((border * 2 + 1, border * 2 + 1), np.uint8)
        outline_alpha = cv2.dilate(alpha_np, kernel, iterations=1)
        stroke_bg = Image.new("RGBA", img_rgba.size, (255, 255, 255, 255))
        stroke_bg.putalpha(Image.fromarray(outline_alpha))
        img_rgba = Image.alpha_composite(stroke_bg, img_rgba)
        
    # --- v15.1 關鍵修正開始 ---
    
    # 2. 【關鍵】找出內容的「緊密邊界框 (Tight BBox)」
    # getbbox() 會回傳非透明區域的最小矩形座標 (left, top, right, bottom)
    bbox = img_rgba.getbbox()
    
    if not bbox:
        # 防呆：如果是全透明圖，回傳空畫布
        return Image.new("RGBA", (370, 320), (0,0,0,0))

    # 3. 緊密裁切 (Crop Tight)
    # 這一步把周圍多餘的透明全部切掉
    tight_img = img_rgba.crop(bbox)
    
    # 4. 應用額外放大倍率 (Apply Zoom to the tight crop)
    if zoom > 1.0:
        tight_w, tight_h = tight_img.size
        new_tight_w = int(tight_w * zoom)
        new_tight_h = int(tight_h * zoom)
        # 使用高品質縮放
        tight_img = tight_img.resize((new_tight_w, new_tight_h), Image.Resampling.LANCZOS)
        
    # --- 最終構圖 ---
    
    # 5. 準備要貼上的最終影像 (保持比例縮放到能塞進 370x320)
    final_sticker_content = tight_img.copy()
    final_sticker_content.thumbnail((370, 320), Image.Resampling.LANCZOS)
    
    # 6. 建立最終透明畫布並置中貼上
    final_bg = Image.new("RGBA", (370, 320), (0, 0, 0, 0))
    fw, fh = final_sticker_content.size
    
    # 計算置中位置
    left = (370 - fw) // 2
    top = (320 - fh) // 2
    
    # 應用垂直位移 (Offset Y) - 移動最終貼圖在畫布上的位置
    top = top + offset_y
    
    # 貼上
    final_bg.paste(final_sticker_content, (left, top))
    
    return final_bg

def process_image(image_pil, slice_strategy, padding, sens, prot, border, d_str, ero, soft, dilation_val, zoom, off_y):
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    h_full, w_full, _ = img_cv.shape
    img_cv = img_cv[10:h_full-10, 10:w_full-10]
    results = []
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
                sticker = extract_content_smart_v14(chunk, sens, prot, d_str, ero, soft, dilation_val)
                if sticker:
                    final = add_stroke_and_resize(sticker, border, zoom, off_y)
                    results.append(final)
    elif "純智慧視覺" in slice_strategy:
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
        st.toast("🚀 啟動 v15.1 修正引擎...", icon="✨")
        st.session_state.processed_stickers = []
        st.session_state.original_images = []
        
        with st.status("正在進行 AI 切割、緊密裁切與二次構圖...", expanded=True) as status:
            prog = st.progress(0)
            for i, f in enumerate(uploaded_files):
                img = Image.open(f).convert("RGB")
                st.session_state.original_images.append((f.name, img))
                
                res = process_image(
                    img, slice_mode, grid_padding, 
                    gs_sensitivity, highlight_protection, border_thickness,
                    despill_level, mask_erode, edge_softness, dilation_strength,
                    zoom_factor, offset_y
                )
                st.session_state.processed_stickers.extend(res)
                prog.progress((i+1)/len(uploaded_files))
            
            if st.session_state.processed_stickers:
                status.update(label="✅ 處理完成", state="complete", expanded=False)
                st.success(f"🎉 成功產出 {len(st.session_state.processed_stickers)} 張貼圖！(已自動去除多餘透明邊框)")
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
            
    st.download_button("📦 下載 v15.1 懶人包", buf.getvalue(), "SarahDad_v15.1.zip", "application/zip", type="primary")
