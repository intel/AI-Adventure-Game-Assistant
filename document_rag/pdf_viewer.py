import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PySide6.QtPdfWidgets import QPdfView
from PySide6.QtPdf import QPdfDocument
from PySide6.QtCore import QPointF,QTimer

class PDFViewer(QMainWindow):
    def __init__(self, pdf_path, start_page=0):
        super().__init__()
        self.setWindowTitle("PDF Viewer")

        # Create a QPdfDocument
        self.document = QPdfDocument(self)
        self.document.load(pdf_path)

        # Create a QPdfView to display the document
        self.pdf_view = QPdfView(self)
        self.pdf_view.setDocument(self.document)
        print("page count",self.document.pageCount())

        # Set the initial page
        self.pdf_view.setPageMode(QPdfView.PageMode.MultiPage)
        self.pdf_view.setZoomMode(QPdfView.ZoomMode.FitToWidth)
        if 0 <= start_page < self.document.pageCount():
            QTimer.singleShot(100, lambda: self.pdf_view.pageNavigator().jump(start_page, QPointF(0, 0)))
        #self.pdf_view.setPage(start_page)

        # Layout setup
        central_widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.pdf_view)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    pdf_path = "Wizards_of_the_Coast_Essentials_Kit.pdf"  # Replace with your PDF file path
    start_page = 50  # Open at page 5 (zero-based index)
    
    viewer = PDFViewer(pdf_path, start_page-1)
    viewer.resize(1000, 800)
    viewer.show()
    
    sys.exit(app.exec())