#!/usr/bin/env python3
"""
SAM Segmentation Application
Приложение для интерактивной сегментации изображений
"""

import sys
import os
import cv2

# Добавляем путь к проекту
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication
from gui.main_window import MainWindow
from sam_engine import SAMEngine

def main():
    """Точка входа в приложение"""
    
    print("=" * 50)
    print("🎯 SAM Segmentation Application")
    print("=" * 50)
    
    # Создаём QApplication
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Красивый стиль
    
    # Инициализируем SAM
    weights_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "weights",
        "mobile_sam.pt"
    )
    
    if not os.path.exists(weights_path):
        print(f"❌ Файл с весами не найден: {weights_path}")
        print("💡 Скачайте его командой:")
        print("   wget https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt")
        print(f"   в папку: {os.path.dirname(weights_path)}")
        sys.exit(1)
    
    try:
        sam_engine = SAMEngine(weights_path=weights_path)
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        sys.exit(1)
    
    # Создаём главное окно
    window = MainWindow(sam_engine)
    window.show()
    
    print("✅ Приложение запущено!")
    print("📝 Инструкция:")
    print("   1. Нажмите 'Загрузить изображение'")
    print("   2. Кликните на объект, который хотите выделить")
    print("   3. Наслаждайтесь результатом! 🎉")
    print("=" * 50)
    
    # Запускаем event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main()