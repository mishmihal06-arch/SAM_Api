from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PySide6.QtGui import QPixmap, QImage, QPainter  # <-- ДОБАВЛЕНО: QPainter
from PySide6.QtCore import Signal, Qt
import numpy as np
import cv2

class ImageCanvas(QGraphicsView):
    """
    Виджет для отображения изображения и обработки кликов
    """
    
    # Сигнал: когда пользователь кликнул (x, y)
    point_clicked = Signal(int, int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Создаём сцену
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        # Элемент для отображения картинки
        self.image_item = QGraphicsPixmapItem()
        self.scene.addItem(self.image_item)
        
        # Текущее изображение (numpy array)
        self.current_image = None
        self.mask_overlay = None
        
        # Настройки отображения (ИСПРАВЛЕНО для PySide6)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        
        # Разрешаем прокрутку, если картинка большая
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
    
    def load_image(self, image_path: str):
        """Загружаем изображение из файла"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.current_image = image_rgb
        self._display_image(image_rgb)
    
    def set_image_array(self, image: np.ndarray):
        """Устанавливаем изображение из numpy массива"""
        self.current_image = image
        self._display_image(image)
    
    def _display_image(self, image: np.ndarray):
        """Отображаем изображение на сцене"""
        height, width, channel = image.shape
        
        bytes_per_line = 3 * width
        qimage = QImage(
            image.data, width, height, bytes_per_line,
            QImage.Format.Format_RGB888  # ИСПРАВЛЕНО для PySide6
        )
        
        pixmap = QPixmap.fromImage(qimage)
        self.image_item.setPixmap(pixmap)
        
        self.scene.setSceneRect(0, 0, width, height)
        self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio) # ИСПРАВЛЕНО
    
    def update_mask(self, mask: np.ndarray):
        """Диагностическая версия отображения"""
        if self.current_image is None:
            print("⚠️ Нет изображения!")
            return
        
        print("\n🎨 update_mask вызван:")
        print(f"   mask shape: {mask.shape}")
        print(f"   mask dtype: {mask.dtype}")
        print(f"   mask min/max: {mask.min():.4f} / {mask.max():.4f}")
        
        # Проверяем тип маски
        if mask.dtype in [np.float32, np.float64]:
            print("   ⚠️ Маска FLOAT — применяем порог 0.5")
            binary_mask = mask > 0.5
        else:
            print("   ✓ Маска уже бинарная")
            binary_mask = mask > 0
        
        mask_pixels = int(np.sum(binary_mask))
        total = binary_mask.shape[0] * binary_mask.shape[1]
        percent = (mask_pixels / total) * 100
        
        print(f"   Бинарная маска: {mask_pixels} пикселей ({percent:.2%})")
        
        if mask_pixels == 0:
            print("   ❌ Маска ПУСТАЯ!")
            return
        
        if percent < 0.1:
            print(f"   ⚠️ Маска ОЧЕНЬ маленькая ({percent:.3f}%) — SAM выделил только точку клика!")
        
        # Создаём цветную маску
        overlay = self.current_image.copy()
        overlay[binary_mask] = [255, 0, 0]  # Красный
        
        # Смешиваем
        blended = cv2.addWeighted(overlay, 0.5, self.current_image, 0.5, 0)
        
        self._display_image(blended)
        print("   ✅ Маска отображена!\n")
    
    def reset_view(self):
        """Сбрасываем к оригинальному изображению"""
        if self.current_image is not None:
            self._display_image(self.current_image)
    
    def mousePressEvent(self, event):
        """Обрабатываем клик мыши с правильным пересчётом координат"""
        if self.current_image is None:
            return
        
        # Получаем координаты клика на сцене
        scene_pos = self.mapToScene(event.pos())
        
        # Получаем размер сцены (оригинальное изображение)
        scene_rect = self.scene.sceneRect()
        
        # Пересчитываем координаты в пиксели оригинального изображения
        if scene_rect.width() > 0 and scene_rect.height() > 0:
            x = int(scene_pos.x() * self.current_image.shape[1] / scene_rect.width())
            y = int(scene_pos.y() * self.current_image.shape[0] / scene_rect.height())
        else:
            x = int(scene_pos.x())
            y = int(scene_pos.y())
        
        # Проверяем, что клик внутри изображения
        h, w = self.current_image.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            print(f"🖱️ Клик: окно=({event.pos().x()}, {event.pos().y()}), оригинал=({x}, {y})")
            
            # Отправляем сигнал с координатами
            self.point_clicked.emit(x, y)
            
            # НЕ РИСУЕМ ТОЧКУ СРАЗУ - пусть main_window сам решит, что рисовать
            # self._draw_point(x, y)  # <-- ЗАКОММЕНТИРОВАНО
        
        super().mousePressEvent(event)
    
    def _draw_point(self, x: int, y: int):
        """Рисуем точку в месте клика"""
        if self.current_image is None:
            return
        
        temp_image = self.current_image.copy()
        cv2.circle(temp_image, (x, y), 5, (255, 0, 0), -1)
        cv2.circle(temp_image, (x, y), 8, (0, 255, 0), 2)
        
        self._display_image(temp_image)
    
    def get_current_image(self) -> np.ndarray:
        """Возвращаем текущее изображение"""
        return self.current_image