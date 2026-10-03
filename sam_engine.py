import torch
import numpy as np
from mobile_sam import sam_model_registry, SamPredictor
import cv2

class SAMEngine:
    """Движок для сегментации на основе MobileSAM"""
    
    def __init__(self, weights_path: str = "weights/mobile_sam.pt"):
        """
        Инициализация модели
        
        Args:
            weights_path: путь к файлу с весами
        """
        print("🔄 Загрузка MobileSAM...")
        
        # Проверяем, есть ли CUDA
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"💻 Используем устройство: {self.device}")
        
        # Загружаем модель
        model_type = "vit_t"  # tiny version для MobileSAM
        sam = sam_model_registry[model_type](checkpoint=weights_path)
        sam.to(device=self.device)
        sam.eval()
        
        # Создаём предсказатель
        self.predictor = SamPredictor(sam)
        self.current_image = None
        
        print("✅ MobileSAM загружен!")
    
    def set_image(self, image: np.ndarray):
        """
        Устанавливаем изображение для сегментации
        
        Args:
            image: изображение в формате numpy array (H, W, 3)
        """
        self.current_image = image
        self.predictor.set_image(image)
        print(f"🖼️ Изображение установлено: {image.shape}")
    
    def predict_by_point(self, x: int, y: int) -> tuple:
        """
        Сегментация по точке (клику мыши)
        
        Args:
            x: координата X
            y: координата Y
            
        Returns:
            mask: бинарная маска (H, W)
            confidence: уверенность модели
        """
        if self.current_image is None:
            raise ValueError("Сначала установите изображение через set_image()")
        
        # Предсказываем маску
        masks, scores, logits = self.predictor.predict(
            point_coords=np.array([[x, y]]),
            point_labels=np.array([1]),  # 1 = положительная точка
            multimask_output=True
        )
        
        # Выбираем лучшую маску (с максимальной уверенностью)
        best_idx = np.argmax(scores)
        return masks[best_idx], scores[best_idx]
    
    def predict_by_box(self, x1: int, y1: int, x2: int, y2: int) -> tuple:
        """
        Сегментация по bounding box
        
        Args:
            x1, y1: верхний левый угол
            x2, y2: нижний правый угол
            
        Returns:
            mask: бинарная маска
            confidence: уверенность
        """
        if self.current_image is None:
            raise ValueError("Сначала установите изображение")
        
        masks, scores, logits = self.predictor.predict(
            box=np.array([x1, y1, x2, y2]),
            multimask_output=True
        )
        
        best_idx = np.argmax(scores)
        return masks[best_idx], scores[best_idx]
    
    def create_colored_mask(self, mask: np.ndarray, color: tuple = (255, 0, 0), alpha: float = 0.5) -> np.ndarray:
        """
        Создаём цветную полупрозрачную маску для отображения
        
        Args:
            mask: бинарная маска (H, W)
            color: цвет в формате BGR (для OpenCV)
            alpha: прозрачность (0 = полностью прозрачная, 1 = непрозрачная)
            
        Returns:
            overlay: изображение с наложенной маской
        """
        if self.current_image is None:
            raise ValueError("Нет текущего изображения")
        
        # Создаём цветную маску
        colored_mask = np.zeros_like(self.current_image)
        colored_mask[mask] = color
        
        # Смешиваем с оригинальным изображением
        overlay = cv2.addWeighted(colored_mask, alpha, self.current_image, 1 - alpha, 0)
        
        # Для областей вне маски оставляем оригинал
        result = self.current_image.copy()
        result[mask] = overlay[mask]
        
        return result