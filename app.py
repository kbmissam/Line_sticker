import streamlit as st
from PIL import Image, ImageDraw, ImageOps, ImageFilter
from rembg import remove
import io
import zipfile
import numpy as np
import cv2
import math

# --- 頁面設定 ---
st.set_page_config(page_title="莎拉爸貼圖神器 v13.0", page_icon="🐴", layout="wide")
st.title("🐴 莎拉爸貼圖神器 v13.0 (影視級去背版)")
st.markdown("🚀 **v13.0 更新**：保留 v12 中心鎖定切割，新增 **「Despill 去綠邊演算法」** 與 **「邊緣羽化」**，徹底消除綠色殘留與鋸齒。")

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
    
    st.sidebar.markdown("##### 🧼 淨化工具 (關鍵)")
    despill_level = st.sidebar.slider("🧪 去綠邊強度 (Despill)", 0.0, 1.0, 0.8, help="將邊緣的綠色反光轉為自然灰色。解決「綠色光暈」的神器。")
    mask_erode = st.sidebar.slider("🤏 遮罩內縮 (Choke)", 0, 5, 1, help="將邊緣向內吃掉 X 像素，物理去除綠邊。")
    edge_softness = st.sidebar.slider("☁️ 邊緣羽化 (Softness)", 0, 10, 3, help="讓邊緣平滑，消除鋸齒。")

st.sidebar.header("3. 裝飾與切割")
border_thickness = st.sidebar.slider("⚪ 白邊厚度", 0, 20, 8)
slice_mode = st.sidebar.radio("切割策略", ("🎯 中心鎖定智慧切割 (保留 v12)", "🧠 純智慧視覺", "📏 自由多線"))

grid_padding = 50
if "中心鎖定" in slice_mode:
    grid_padding = st.sidebar.slider("↔️ 抓取寬容度 (Padding)", 10, 150, 50, help="建議 40-60，確保手腳不被切斷。")

# --- 核心演算法區 ---

def apply_despill(img_bgr, strength=0.8):
    """
    專業級 Despill 演算法：
    當像素的綠色通道 (G) 大於 紅(R) 和 藍(B) 時，
    強制將 G 壓低到 R 和 B 的平均值，從而消除綠色色偏。
    """
    img_float = img_bgr.astype(np.float32)
    b, g, r = cv2.split(img_float)
    
    # 計算 Despill 目標值 (取 R 和 B 的平均)
    rb_avg = (r + b) / 2.0
    
    # 找出綠色溢出的地方 (G > RB_Average)
    spill_mask = g > rb_avg
    
    # 強制壓低綠色
    g[spill_mask] = g[spill_mask] * (1 - strength) + rb_avg[spill_mask] * strength
    
    # 合併回 BGR
    despilled = cv2.merge([b, g, r])
    return np.clip(despilled, 0, 255).astype(np.uint8)

def get_pro_matte(chunk_cv, sensitivity, protect, erode_iter, softness):
    """生成高品質 Alpha 遮罩"""
    # 1. 基礎 HSV 遮罩 (抓背景)
    hsv = cv2.cvtColor(chunk_cv, cv2.COLOR_BGR2HSV)
    sat_threshold = 140 - int(sensitivity * 0.9)
    lower_green = np.array([35, sat_threshold, 40])
    upper_green = np.array([85, 255, 255])
    bg_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # 2. 亮部保護
    if protect > 0:
        s_max = int(protect * 0.8)
        v_min = 255 - int(protect * 1.5)
        lower_white = np.array([0, 0, v_min])
        upper_white = np.array([180, s_max, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        bg_mask = cv2.bitwise_and(bg_mask, cv2.bitwise_not(white_mask))
    
    # 3. 反轉為前景遮罩 (0=背景, 255=前景)
    fg_mask = cv2.bitwise_not(bg_mask)
    
    # 4. 物理內縮 (Erode) - 吃掉綠邊
    if erode_iter > 0:
        kernel = np.ones((3,3), np.uint8)
        fg_mask = cv2.erode(fg_mask, kernel, iterations=erode_iter)
        
    # 5. 邊緣羽化 (Gaussian Blur) - 消除鋸齒
    if softness > 0:
        k_size = softness * 2 + 1
        fg_mask = cv2.GaussianBlur(fg_mask, (k_size, k_size), 0)
        
    return fg_mask

def extract_content_smart_v13(chunk_cv, sensitivity, protect, d_strength, erode, soft):
    """整合 Despill 與 Center Lock"""
    h, w, _ = chunk_cv.shape
    center_x, center_y = w // 2, h // 2
    
    # 1. 先用簡單遮罩找輪廓 (為了定位中心)
    simple_mask = get_pro_matte(chunk_cv, sensitivity, protect, 0, 0)
    contours, _ = cv2.findContours(simple_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours: return None
    min_area = 500
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    if not valid_contours: return None
    
    # 2. 中心鎖定 (保留 v12 邏輯)
    best_cnt = None
    min_dist = float('inf')
    for cnt in valid_contours:
        x,y,cw,ch = cv2.boundingRect(cnt)
        cX, cY = x + cw//2, y + ch//2
        dist = math.sqrt((cX - center_x)**2 + (cY - center_y)**2)
        if dist < min_dist:
            min_dist = dist
            best_cnt = cnt
            
    if best_cnt is None: return None
    
    # 3. 建立最終的高品質遮罩 (只包含最佳輪廓區域)
    # 先做全圖的高級遮罩
    high_quality_mask = get_pro_matte(chunk_cv, sensitivity, protect, erode, soft)
    
    # 建立一個只包含 best_cnt 的過濾器
    island_filter = np.zeros_like(high_quality_mask)
    # 這裡稍微膨脹輪廓遮罩，以免切到羽化的邊緣
    hull = cv2.convexHull(best_cnt) 
    cv2.drawContours(island_filter, [hull], -1, 255, thickness=cv2.FILLED)
    
    # 交集：(高品質遮罩) AND (中心島嶼位置)
    final_alpha = cv2.bitwise_and(high_quality_mask, island_filter)
    
    # 4. 應用 Despill 去色 (把邊緣綠光變灰)
    if d_strength > 0:
        chunk_clean = apply_despill(chunk_cv, d_strength)
    else:
        chunk_clean = chunk_cv
        
    # 5. 組合 RGBA
    b, g, r = cv2.split(chunk_clean)
    rgba = cv2.merge([r, g, b, final_alpha])
    
    # 6. 裁切
    x, y, cw, ch = cv2.boundingRect(best_cnt)
    # 稍微往外擴一點裁切，保留羽化邊緣
    pad = soft + 2
    x = max(0, x - pad); y = max(0, y - pad)
    cw = min(w - x, cw + pad*2); ch = min(h - y, ch + pad*2)
    
    final_chunk = rgba[y:y+ch, x:x+cw]
    
    return Image.fromarray(final_chunk)

def add_stroke_and_resize(sticker_pil, border):
    # 加白邊
    if border > 0:
        img = sticker_pil.convert("RGBA")
        r, g, b, a = img.split()
        alpha_np = np.array(a)
        
        # 為了讓白邊圓潤，先對 Alpha 做一點 Blur
        # alpha_blur = cv2.GaussianBlur(alpha_np, (3,3), 0)
        
        kernel = np.ones((border * 2 + 1, border * 2 + 1), np.uint8)
        outline_alpha = cv2.dilate(alpha_np, kernel, iterations=1)
        
        stroke_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        stroke_bg.putalpha(Image.fromarray(outline_alpha))
        sticker_pil = Image.alpha_composite(stroke_bg, img)

    # 縮放與置中
    sticker_pil.thumbnail((370, 320), Image.Resampling.LANCZOS)
    final_bg = Image.new("RGBA", (370, 320), (0, 0, 0, 0))
    left = (370 - sticker_pil.width) // 2
    top = (320 - sticker_pil.height) // 2
    final_bg.paste(sticker_pil, (left, top))
    return final_bg

def process_image(image_pil, slice_strategy, padding, sens, prot, border, d_str, ero, soft):
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    # Pre-crop
    h_full, w_full, _ = img_cv.shape
    img_cv = img_cv[10:h_full-10, 10:w_full-10]
    
    results = []
    
    if "中心鎖定" in slice_strategy:
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
                
                # 呼叫 v13 升級版函數
                sticker = extract_content_smart_v13(chunk, sens, prot, d_str, ero, soft)
                
                if sticker:
                    final = add_stroke_and_resize(sticker, border)
                    results.append(final)
        
        return results
    else:
        return []

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
        st.toast("🚀 啟動影視級去背引擎...", icon="✨")
        st.session_state.processed_stickers = []
        st.session_state.original_images = []
        
        with st.status("正在進行 Despill 與 Alpha Matting 運算...", expanded=True) as status:
            prog = st.progress(0)
            for i, f in enumerate(uploaded_files):
                img = Image.open(f).convert("RGB")
                st.session_state.original_images.append((f.name, img))
                
                res = process_image(
                    img, slice_mode, grid_padding, 
                    gs_sensitivity, highlight_protection, border_thickness,
                    despill_level, mask_erode, edge_softness
                )
                st.session_state.processed_stickers.extend(res)
                prog.progress((i+1)/len(uploaded_files))
            
            if st.session_state.processed_stickers:
                status.update(label="✅ 畫質優化完成", state="complete", expanded=False)
                st.success(f"🎉 成功產出 {len(st.session_state.processed_stickers)} 張高畫質貼圖！")
            else:
                status.update(label="⚠️ 失敗", state="error")
                st.error("未偵測到貼圖，請調整參數。")

# 預覽區 (保持不變)
if st.session_state.processed_stickers:
    st.divider()
    st.header("🖼️ 成果預覽")
    
    opts = [f"{i+1:02d}" for i in range(len(st.session_state.processed_stickers))]
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.subheader("設定縮圖")
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
        
        bm, bt = io.BytesIO(), io.BytesIO()
        m_img.save(bm, "PNG"); zf.writestr("main.png", bm.getvalue())
        t_img.save(bt, "PNG"); zf.writestr("tab.png", bt.getvalue())
            
    st.download_button("📦 下載 v13.0 懶人包", buf.getvalue(), "SarahDad_v13.0.zip", "application/zip", type="primary")
