"""
LINE Sticker Batch Processor - Image Processing Module
處理影像分割、去背、裁剪和調整大小
"""

from PIL import Image
import numpy as np
from rembg import remove
import io
from typing import List, Tuple, Optional


class StickerProcessor:
    """貼紙處理器類別"""
    
    # LINE 貼紙最大尺寸
    MAX_WIDTH = 370
    MAX_HEIGHT = 320
    
    def __init__(self, grid_cols: int = 6, grid_rows: int = 5):
        """
        初始化貼紙處理器
        
        Args:
            grid_cols: 網格列數
            grid_rows: 網格行數
        """
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows
        self.processed_stickers: List[Image.Image] = []
    
    def slice_image(self, image: Image.Image) -> List[Image.Image]:
        """
        根據網格設定分割影像
        
        Args:
            image: 輸入影像
            
        Returns:
            分割後的影像列表
        """
        width, height = image.size
        cell_width = width // self.grid_cols
        cell_height = height // self.grid_rows
        
        slices = []
        
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                left = col * cell_width
                top = row * cell_height
                right = left + cell_width
                bottom = top + cell_height
                
                # 確保邊界不超出影像
                right = min(right, width)
                bottom = min(bottom, height)
                
                cell = image.crop((left, top, right, bottom))
                slices.append(cell)
        
        return slices
    
    def remove_background(self, image: Image.Image) -> Image.Image:
        """
        使用 rembg 移除背景
        
        Args:
            image: 輸入影像
            
        Returns:
            去背後的影像
        """
        try:
            # 將 PIL Image 轉換為 bytes
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            # 使用 rembg 移除背景
            output = remove(img_bytes.read())
            
            # 將結果轉換回 PIL Image
            result = Image.open(io.BytesIO(output)).convert('RGBA')
            return result
        except Exception as e:
            print(f"背景移除失敗: {e}")
            # 如果失敗，返回原始影像轉換為 RGBA
            return image.convert('RGBA')
    
    def trim_transparency(self, image: Image.Image) -> Image.Image:
        """
        裁剪透明邊框
        
        Args:
            image: 輸入影像 (應為 RGBA)
            
        Returns:
            裁剪後的影像
        """
        # 確保影像是 RGBA
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # 獲取 alpha 通道
        alpha = image.split()[-1]
        
        # 找到非透明區域的邊界
        bbox = alpha.getbbox()
        
        if bbox is None:
            # 如果整個影像都是透明的，返回原始影像
            return image
        
        # 裁剪影像
        return image.crop(bbox)
    
    def resize_to_line_spec(self, image: Image.Image) -> Image.Image:
        """
        調整影像大小以符合 LINE 貼紙規格 (370x320)
        同時保持寬高比，使用高品質重新採樣
        
        Args:
            image: 輸入影像
            
        Returns:
            調整大小後的影像
        """
        # 確保影像是 RGBA
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # 計算縮放比例以符合最大尺寸
        width, height = image.size
        
        # 計算需要的縮放比例
        scale = min(self.MAX_WIDTH / width, self.MAX_HEIGHT / height)
        
        # 如果影像已經小於最大尺寸，不放大
        if scale >= 1:
            return image
        
        # 計算新尺寸
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # 使用 LANCZOS 高品質重新採樣
        resized = image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )
        
        return resized
    
    def process_sticker(self, image: Image.Image) -> Optional[Image.Image]:
        """
        處理單個貼紙：去背 -> 裁剪 -> 調整大小
        
        Args:
            image: 輸入影像
            
        Returns:
            處理後的影像，如果失敗則返回 None
        """
        try:
            # 檢查影像是否為空（全白或全黑）
            if self._is_empty_image(image):
                return None
            
            # 步驟 1: 移除背景
            no_bg = self.remove_background(image)
            
            # 步驟 2: 裁剪透明邊框
            trimmed = self.trim_transparency(no_bg)
            
            # 檢查裁剪後是否還有內容
            if trimmed.size[0] < 10 or trimmed.size[1] < 10:
                return None
            
            # 步驟 3: 調整大小
            resized = self.resize_to_line_spec(trimmed)
            
            return resized
        except Exception as e:
            print(f"貼紙處理失敗: {e}")
            return None
    
    def _is_empty_image(self, image: Image.Image, threshold: int = 240) -> bool:
        """
        檢查影像是否為空（大部分像素為白色或接近白色）
        
        Args:
            image: 輸入影像
            threshold: 白色閾值 (0-255)
            
        Returns:
            如果影像為空則返回 True
        """
        try:
            # 轉換為 RGB
            if image.mode != 'RGB':
                img_rgb = image.convert('RGB')
            else:
                img_rgb = image
            
            # 轉換為 numpy 陣列
            arr = np.array(img_rgb)
            
            # 計算平均亮度
            avg_brightness = arr.mean()
            
            # 如果平均亮度接近白色，視為空影像
            return avg_brightness > threshold
        except Exception:
            return False
    
    def process_batch(self, image: Image.Image, progress_callback=None) -> List[Image.Image]:
        """
        批量處理貼紙
        
        Args:
            image: 輸入的貼紙表單影像
            progress_callback: 進度回調函數 (接收當前進度和總數)
            
        Returns:
            處理後的貼紙影像列表
        """
        # 分割影像
        slices = self.slice_image(image)
        total = len(slices)
        
        self.processed_stickers = []
        
        for idx, slice_img in enumerate(slices):
            # 更新進度
            if progress_callback:
                progress_callback(idx + 1, total)
            
            # 處理貼紙
            processed = self.process_sticker(slice_img)
            
            # 只添加非空的貼紙
            if processed is not None:
                self.processed_stickers.append(processed)
        
        return self.processed_stickers
    
    def get_processed_stickers(self) -> List[Image.Image]:
        """
        獲取已處理的貼紙列表
        
        Returns:
            處理後的貼紙影像列表
        """
        return self.processed_stickers
