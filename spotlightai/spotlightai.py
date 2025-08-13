import sys
import logging
import threading
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QWidget, QLineEdit, QVBoxLayout, 
                            QTextEdit, QDesktopWidget, QHBoxLayout, QPushButton)
from PyQt5.QtCore import Qt, QPropertyAnimation, QEasingCurve, QRect, pyqtSignal, QObject
from PyQt5.QtGui import QPainter, QColor, QFont, QTextCursor
from RealtimeSTT import AudioToTextRecorder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TranscriptionHandler(QObject):
    new_text_signal = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.full_text = ""
    
    def process_text(self, text):
        self.new_text_signal.emit(text)
        self.full_text = text


class SpotlightAI(QWidget):
    def __init__(self):
        super().__init__()
        self.recorder = None
        self.transcription_handler = TranscriptionHandler()
        self.transcription_handler.new_text_signal.connect(self.update_transcription)
        self.recording = False
        self.initUI()
        
    def initUI(self):
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)

        # Create a horizontal layout for microphone button and text display
        top_layout = QHBoxLayout()
        
        # Microphone button
        self.mic_button = QPushButton("🎙️", self)
        self.mic_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(60, 60, 60, 200);
                border: none;
                border-radius: 25px;
                padding: 10px;
                color: white;
                font-size: 20px;
                min-width: 50px;
                max-width: 50px;
                min-height: 50px;
                max-height: 50px;
            }
            QPushButton:pressed {
                background-color: rgba(0, 150, 255, 200);
            }
        """)
        self.mic_button.clicked.connect(self.toggle_recording)
        top_layout.addWidget(self.mic_button)

        # Real-time transcription display
        self.transcription_display = QLineEdit(self)
        self.transcription_display.setPlaceholderText("Click microphone to start speech recognition...")
        self.transcription_display.setStyleSheet("""
            QLineEdit {
                background-color: rgba(60, 60, 60, 200);
                border: none;
                border-radius: 20px;
                padding: 15px;
                color: white;
                font-size: 16px;
            }
        """)
        self.transcription_display.setReadOnly(True)
        top_layout.addWidget(self.transcription_display, 1)

        layout.addLayout(top_layout)

        # Result area for final transcriptions
        self.result_area = QTextEdit(self)
        self.result_area.setReadOnly(True)
        self.result_area.setStyleSheet("""
            QTextEdit {
                background-color: rgba(60, 60, 60, 200);
                border: none;
                border-radius: 10px;
                padding: 10px;
                color: white;
                font-size: 14px;
            }
        """)
        self.result_area.hide()
        layout.addWidget(self.result_area)

        self.setLayout(layout)

        # Position window
        screen = QDesktopWidget().screenNumber(QDesktopWidget().cursor().pos())
        screen_size = QDesktopWidget().screenGeometry(screen)
        window_width = 800
        window_height = 80
        x = (screen_size.width() - window_width) // 2
        y = screen_size.height() // 4
        self.setGeometry(x, y, window_width, window_height)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor(30, 30, 30, 200))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(self.rect(), 20, 20)

    def toggle_recording(self):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        try:
            self.recording = True
            self.mic_button.setText("⏹")
            self.mic_button.setStyleSheet("""
                QPushButton {
                    background-color: rgba(255, 0, 0, 200);
                    border: none;
                    border-radius: 25px;
                    padding: 10px;
                    color: white;
                    font-size: 20px;
                    min-width: 50px;
                    max-width: 50px;
                    min-height: 50px;
                    max-height: 50px;
                }
            """)
            self.transcription_display.setText("Listening...")
            
            # Initialize recorder with real-time transcription
            self.recorder = AudioToTextRecorder(
                model="tiny",
                language="",
                enable_realtime_transcription=True,
                on_realtime_transcription_update=self.on_realtime_update,
                spinner=False,
                use_microphone=True
            )
            
            # Start manual recording
            self.recorder.start()
            
        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            self.recording = False
            self.mic_button.setText("🎙️")
            self.transcription_display.setText(f"Error: {str(e)}")
            
    def stop_recording(self):
        self.recording = False
        self.mic_button.setText("🎙️")
        self.mic_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(60, 60, 60, 200);
                border: none;
                border-radius: 25px;
                padding: 10px;
                color: white;
                font-size: 20px;
                min-width: 50px;
                max-width: 50px;
                min-height: 50px;
                max-height: 50px;
            }
            QPushButton:pressed {
                background-color: rgba(0, 150, 255, 200);
            }
        """)
        
        if self.recorder:
            try:
                self.recorder.stop()
                final_text = self.recorder.text()
                if final_text and final_text.strip():
                    self.show_final_transcription(final_text)
                self.recorder.shutdown()
            except Exception as e:
                logger.error(f"Error stopping recording: {e}")
            self.recorder = None
        
        self.transcription_display.setText("Click microphone to start recording...")

    def on_realtime_update(self, text):
        # Update the real-time display using signal/slot mechanism
        self.transcription_handler.new_text_signal.emit(text)

    def update_transcription(self, text):
        self.transcription_display.setText(text)

    def on_recording_start(self):
        logger.info("Recording started")

    def on_recording_stop(self):
        logger.info("Recording stopped")
        # Don't auto-stop, let user control it

    def show_final_transcription(self, text):
        if text and text.strip():
            timestamp = datetime.now().strftime("%H:%M:%S")
            final_text = f"[{timestamp}] {text}\n"
            
            if self.result_area.isHidden():
                self.result_area.show()
                self.animate_expand()
            
            cursor = self.result_area.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.insertText(final_text)
            self.result_area.setTextCursor(cursor)
            self.result_area.ensureCursorVisible()

    def animate_expand(self):
        self.animation = QPropertyAnimation(self, b"geometry")
        self.animation.setDuration(300)
        self.animation.setStartValue(self.geometry())
        new_height = 300
        new_geometry = QRect(self.x(), self.y(), self.width(), new_height)
        self.animation.setEndValue(new_geometry)
        self.animation.setEasingCurve(QEasingCurve.OutCubic)
        self.animation.start()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

    def closeEvent(self, event):
        if self.recording:
            self.stop_recording()
        logger.info("Application closing")
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    ex = SpotlightAI()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()