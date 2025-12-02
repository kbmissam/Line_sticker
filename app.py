"""
LINE Sticker Batch Processor - Streamlit 應用程式
使用者介面和主要邏輯
"""

import streamlit as st
from PIL import Image
import io
from image_processor import StickerProcessor
from zip_handler import create_sticker_zip


# 設定頁面配置
st.set_page_config(
    page_title="LINE Sticker Batch Processor",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自訂 CSS 樣式
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #00B900;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 30px;
    }
    .sticker-gallery {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
        gap: 15px;
        margin-top: 20px;
    }
    .info-box {
        background-color: #f0f0f0;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# 標題
st.markdown("<h1 class='main-title'>🎨 LINE Sticker Batch Processor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>自動分割、去背、裁剪和調整貼紙尺寸</p>", unsafe_allow_html=True)

# 側邊欄設定
st.sidebar.header("⚙️ 設定")

# 網格設定
col1, col2 = st.sidebar.columns(2)
with col1:
    grid_cols = st.number_input(
        "網格列數",
        min_value=1,
        max_value=20,
        value=6,
        help="貼紙表單的列數"
    )

with col2:
    grid_rows = st.number_input(
        "網格行數",
        min_value=1,
        max_value=20,
        value=5,
        help="貼紙表單的行數"
    )

# 下載檔名設定
download_filename = st.sidebar.text_input(
    "下載檔名",
    value="Stickers_Done",
    help="ZIP 檔案的名稱 (不含 .zip 副檔名)"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📋 使用說明
1. 上傳包含貼紙表單的影像 (JPG/PNG)
2. 設定網格列數和行數
3. 點擊「開始處理」按鈕
4. 預覽處理結果
5. 下載 ZIP 檔案

### 📏 LINE 貼紙規格
- 最大寬度: 370px
- 最大高度: 320px
- 格式: PNG (透明背景)
""")

# 主要內容區域
st.header("📤 上傳貼紙表單")

uploaded_file = st.file_uploader(
    "選擇影像檔案",
    type=["jpg", "jpeg", "png"],
    help="上傳包含貼紙表單的影像"
)

if uploaded_file is not None:
    # 顯示上傳的影像
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("原始影像")
        st.image(image, use_column_width=True)
        st.info(f"影像尺寸: {image.size[0]}×{image.size[1]} 像素")
    
    with col2:
        st.subheader("處理設定")
        st.write(f"**網格設定**: {grid_cols} 列 × {grid_rows} 行")
        st.write(f"**預期貼紙數**: {grid_cols * grid_rows} 個")
        st.write(f"**下載檔名**: {download_filename}.zip")
    
    # 處理按鈕
    if st.button("🚀 開始處理", use_container_width=True, type="primary"):
        st.session_state.processing = True
        
        # 初始化處理器
        processor = StickerProcessor(grid_cols=grid_cols, grid_rows=grid_rows)
        
        # 進度條
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(current, total):
            progress = current / total
            progress_bar.progress(progress)
            status_text.text(f"處理進度: {current}/{total} ({int(progress*100)}%)")
        
        # 處理批次
        with st.spinner("正在處理貼紙..."):
            processed_stickers = processor.process_batch(image, progress_callback=update_progress)
        
        # 清除進度提示
        progress_bar.empty()
        status_text.empty()
        
        # 儲存到 session state
        st.session_state.processed_stickers = processed_stickers
        st.session_state.processor = processor
        
        if processed_stickers:
            st.success(f"✅ 處理完成！共生成 {len(processed_stickers)} 個貼紙")
        else:
            st.error("❌ 未能生成任何貼紙，請檢查輸入影像")
    
    # 顯示處理結果
    if "processed_stickers" in st.session_state and st.session_state.processed_stickers:
        st.markdown("---")
        st.header("🖼️ 處理結果預覽")
        
        processed_stickers = st.session_state.processed_stickers
        
        # 顯示統計資訊
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("生成的貼紙", len(processed_stickers))
        with col2:
            if processed_stickers:
                avg_width = sum(s.size[0] for s in processed_stickers) / len(processed_stickers)
                st.metric("平均寬度", f"{avg_width:.0f}px")
        with col3:
            if processed_stickers:
                avg_height = sum(s.size[1] for s in processed_stickers) / len(processed_stickers)
                st.metric("平均高度", f"{avg_height:.0f}px")
        
        # 顯示前 12 個貼紙的預覽
        preview_count = min(12, len(processed_stickers))
        st.subheader(f"預覽 (前 {preview_count} 個)")
        
        cols = st.columns(6)
        for idx in range(preview_count):
            with cols[idx % 6]:
                st.image(
                    processed_stickers[idx],
                    use_column_width=True,
                    caption=f"#{idx+1:02d}"
                )
        
        if len(processed_stickers) > preview_count:
            st.info(f"還有 {len(processed_stickers) - preview_count} 個貼紙未顯示")
        
        # 下載按鈕
        st.markdown("---")
        st.header("💾 下載")
        
        # 生成 ZIP 檔案
        zip_data = create_sticker_zip(processed_stickers, download_filename)
        
        st.download_button(
            label=f"📥 下載 {download_filename}.zip",
            data=zip_data,
            file_name=f"{download_filename}.zip",
            mime="application/zip",
            use_container_width=True,
            type="primary"
        )
        
        st.success(f"ZIP 檔案已準備好下載，包含 {len(processed_stickers)} 個貼紙")

else:
    st.info("👈 請在左側上傳貼紙表單影像開始處理")
    
    # 顯示範例資訊
    st.markdown("""
    ### 📝 什麼是貼紙表單？
    
    貼紙表單是一個大影像，其中包含多個貼紙排列成網格。例如：
    - 6 列 × 5 行 = 30 個貼紙
    - 4 列 × 4 行 = 16 個貼紙
    
    ### 🔧 處理流程
    
    1. **分割** - 根據網格設定將大影像分割成小格子
    2. **去背** - 使用 AI 移除每個貼紙的背景
    3. **裁剪** - 自動移除透明邊框
    4. **調整大小** - 縮放至 LINE 貼紙規格 (最大 370×320px)
    5. **打包** - 將所有貼紙打包成 ZIP 檔案
    
    ### ✨ 特點
    
    - ✅ 自動化批量處理
    - ✅ 高品質重新採樣 (LANCZOS)
    - ✅ 智能背景移除
    - ✅ 實時進度顯示
    - ✅ 結果預覽
    - ✅ 一鍵下載
    """)
