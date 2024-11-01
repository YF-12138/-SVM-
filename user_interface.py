import sys

from IPython.external.qt_for_kernel import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QTextEdit, QFileDialog
from PyQt5.QtCore import QRect, QMetaObject
from PyQt5.QtGui import QPixmap
import os
import joblib

import svm  # Assuming svm.img2vector and required methods exist


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(645, 475)

        self.pushButton = QPushButton(Dialog)
        self.pushButton.setGeometry(QRect(230, 340, 141, 41))
        self.pushButton.setAutoDefault(False)
        self.pushButton.setObjectName("pushButton")

        self.label = QLabel(Dialog)
        self.label.setGeometry(QRect(220, 50, 191, 221))
        self.label.setWordWrap(False)
        self.label.setObjectName("label")

        self.textEdit = QTextEdit(Dialog)
        self.textEdit.setGeometry(QRect(220, 280, 191, 41))
        self.textEdit.setObjectName("textEdit")

        self.retranslateUi(Dialog)
        QMetaObject.connectSlotsByName(Dialog)  # Only if needed

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "手写体识别"))
        self.pushButton.setText(_translate("Dialog", "打开图像"))
        self.label.setText(_translate("Dialog", "显示图像"))


class MyWindow(QMainWindow, Ui_Dialog):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.openImage)

    def openImage(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图像", "img_test")
        png = QPixmap(imgName).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(png)
        self.textEdit.setText(imgName)

        model_path = os.path.join(sys.path[0], 'svm.model')
        clf = joblib.load(model_path)
        dataMat = svm.img2vector(imgName)
        preResult = clf.predict(dataMat)

        self.textEdit.setReadOnly(True)
        self.textEdit.setStyleSheet("color: red")
        self.textEdit.setAlignment(QtCore.Qt.AlignCenter)
        self.textEdit.setFontPointSize(9)
        self.textEdit.setText("预测的结果是:")
        self.textEdit.append(str(preResult[0]))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()
    sys.exit(app.exec_())