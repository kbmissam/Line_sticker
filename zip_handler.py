"""
ZIP 檔案處理模組
用於將處理後的貼紙打包成 ZIP 檔案
"""

import zipfile
import io
from PIL import Image
from typing import List


def create_sticker_zip(stickers: List[Image.Image], filename: str = "Stickers_Done") -> bytes:
    """
    將處理後的貼紙打包成 ZIP 檔案
    
    Args:
        stickers: 貼紙影像列表
        filename: ZIP 檔案名稱 (不含副檔名)
        
    Returns:
        ZIP 檔案的 bytes 內容
    """
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for idx, sticker in enumerate(stickers, 1):
            # 生成序號檔名 (01.png, 02.png, ...)
            sticker_filename = f"{idx:02d}.png"
            
            # 將貼紙轉換為 bytes
            img_buffer = io.BytesIO()
            sticker.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            # 添加到 ZIP 檔案
            zip_file.writestr(sticker_filename, img_buffer.getvalue())
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()
