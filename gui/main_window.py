from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox,
    QStatusBar, QToolBar
)
from PySide6.QtCore import Qt
from .image_canvas import ImageCanvas
import os
import cv2

class MainWindow(QMainWindow):
    """Главное окно приложения"""
    
    def __init__(self, sam_engine):
        super().__init__()
        
        self.sam_engine = sam_engine
        self.current_image_path = None
        
        self._init_ui()
        self._create_toolbar()
        self._connect_signals()
    
    def _init_ui(self):
        """Инициализация интерфейса"""
        self.setWindowTitle("🎯 SAM Segmentation App")
        self.setMinimumSize(1024, 768)
        
        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Основной layout
        layout = QVBoxLayout(central_widget)
        
        # Холст с изображением
        self.canvas = ImageCanvas()
        layout.addWidget(self.canvas, stretch=1)
        
        # Панель управления
        control_panel = self._create_control_panel()
        layout.addWidget(control_panel)
        
        # Статус бар
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Готов к работе. Загрузите изображение.")
    
    def _create_control_panel(self) -> QWidget:
        """Создаём панель управления"""
        panel = QWidget()
        layout = QHBoxLayout(panel)
        
        # Кнопка загрузки
        self.btn_load = QPushButton("📂 Загрузить изображение")
        layout.addWidget(self.btn_load)
        
        # Кнопка сброса
        self.btn_reset = QPushButton("🔄 Сбросить маску")
        layout.addWidget(self.btn_reset)
        
        # Информация
        self.info_label = QLabel("Кликните на объект для сегментации")
        self.info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.info_label, stretch=1)
        
        return panel
    
    def _create_toolbar(self):
        """Создаём панель инструментов"""
        toolbar = QToolBar("Основные инструменты")
        self.addToolBar(toolbar)
        
        # Кнопка загрузки файла
        load_action = toolbar.addAction("📂 Открыть")
        load_action.triggered.connect(self.load_image)
        
        # Кнопка сохранения
        save_action = toolbar.addAction("💾 Сохранить маску")
        save_action.triggered.connect(self.save_mask)
        
        toolbar.addSeparator()
        
        # Информация о модели
        device = self.sam_engine.device
        toolbar.addWidget(QLabel(f"🖥️ {device.upper()}"))
    
    def _connect_signals(self):
        """Подключаем сигналы"""
        # Кнопки
        self.btn_load.clicked.connect(self.load_image)
        self.btn_reset.clicked.connect(self.reset_mask)
        
        # Клик по изображению
        self.canvas.point_clicked.connect(self.on_point_click)
    
    def load_image(self):
        """Загружаем изображение через диалог"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите изображение",
            os.path.expanduser("~"),
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_path:
            try:
                self.canvas.load_image(file_path)
                self.current_image_path = file_path
                
                # Устанавливаем в SAM
                image = self.canvas.get_current_image()
                self.sam_engine.set_image(image)
                
                self.status_bar.showMessage(f"Загружено: {os.path.basename(file_path)}")
                
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить изображение:\n{e}")
    
    def on_point_click(self, x: int, y: int):
        print(f"📍 Получен клик: x={x}, y={y}")
        
        if self.sam_engine.current_image is None:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите изображение!")
            return
        
        try:
            print(f"🔄 Начинаю сегментацию...")
            self.status_bar.showMessage(f" Сегментация... точка ({x}, {y})")
            
            # Получаем маску
            mask, confidence = self.sam_engine.predict_by_point(x, y)
            
            print(f"✅ Маска получена! Размер: {mask.shape}, Уверенность: {confidence:.3f}")
            
            # Показываем маску
            self.canvas.update_mask(mask)
            
            # РИСУЕМ ТОЧКУ ПОСЛЕ МАСКИ (добавь это)
            self.canvas._draw_point(x, y)
            
            self.status_bar.showMessage(
                f"✅ Сегментация завершена! Уверенность: {confidence:.2%}"
            )
            
        except Exception as e:
            print(f"❌ ОШИБКА СЕГМЕНТАЦИИ: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Ошибка", f"Ошибка сегментации:\n{e}")
            self.status_bar.showMessage("❌ Ошибка сегментации")
        
    def reset_mask(self):
        """Сбрасываем маску к оригинальному изображению"""
        self.canvas.reset_view()
        self.status_bar.showMessage("Маска сброшена")
    

    def save_mask(self):
        """Сохраняем маску в файл"""
        # Получаем текущее изображение с canvas
        image = self.canvas.get_current_image()
        
        if image is None:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите изображение!")
            return
        
        # Предлагаем сохранить
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить результат",
            os.path.expanduser("~/sam_segmentation_app/datasets/result.png"),
            "PNG Images (*.png);;JPEG Images (*.jpg)"
        )
        
        if save_path:
            try:
                # Сохраняем изображение через OpenCV
                # OpenCV требует BGR, а у нас RGB — конвертируем
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, image_bgr)
                
                QMessageBox.information(
                    self, 
                    "Готово", 
                    f"Изображение сохранено:\n{save_path}"
                )
                self.status_bar.showMessage(f" Сохранено: {save_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить:\n{e}")