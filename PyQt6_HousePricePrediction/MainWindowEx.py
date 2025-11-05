import os
import traceback
import pandas as pd
import numpy as np
import pickle
import datetime
from PyQt6.QtWidgets import (
    QFileDialog, QMessageBox, QTableWidgetItem, QMainWindow
)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from MainWindow import Ui_MainWindow


class MainWindowEx(Ui_MainWindow):
    def __init__(self):
        super().__init__()
        # Dữ liệu mô hình
        self.df = None
        self.lm = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def setupUi(self, MainWindow: QMainWindow):
        super().setupUi(MainWindow)
        self.MainWindow = MainWindow

        # Gán sự kiện cho nút
        self.btnPickDataset.clicked.connect(self.do_pick_data)
        self.btnViewDataset.clicked.connect(self.do_view_dataset)
        self.btnTrain.clicked.connect(self.do_train)
        self.btnEvaluate.clicked.connect(self.do_evaluation)
        self.btnSaveModel.clicked.connect(self.do_save_model)
        self.btnLoadModel.clicked.connect(self.do_load_model)
        self.btnPredict.clicked.connect(self.do_prediction)
        self.comboModels.currentTextChanged.connect(self.on_model_changed)

        # Dữ liệu mặc định
        self.lineDataset.setText("dataset/USA_Housing.csv")

        # Khi khởi động, load danh sách model có sẵn
        self.refresh_model_list()

    def show(self):
        self.MainWindow.show()

    # ========== Chức năng chính ==========
    def do_pick_data(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.MainWindow,
            "Choose dataset",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            self.lineDataset.setText(file_path)

    def do_view_dataset(self):
        try:
            df = pd.read_csv(self.lineDataset.text())
            self.tableResults.setColumnCount(len(df.columns))
            self.tableResults.setRowCount(len(df))
            self.tableResults.setHorizontalHeaderLabels(df.columns)

            for i in range(len(df)):
                for j in range(len(df.columns)):
                    item = QTableWidgetItem(str(df.iloc[i, j]))
                    self.tableResults.setItem(i, j, item)

            QMessageBox.information(self.MainWindow, "Info", "Dataset loaded successfully.")
        except Exception as e:
            QMessageBox.critical(self.MainWindow, "Error", str(e))

    def do_train(self):
        try:
            ratio = self.spinTrainRate.value() / 100
            self.df = pd.read_csv(self.lineDataset.text())

            self.X = self.df[['Avg. Area Income', 'Avg. Area House Age',
                              'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms',
                              'Area Population']]
            self.y = self.df['Price']

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=1 - ratio, random_state=101
            )

            self.lm = LinearRegression()
            self.lm.fit(self.X_train, self.y_train)

            QMessageBox.information(self.MainWindow, "Info", "Training completed successfully.")
        except Exception as e:
            QMessageBox.critical(self.MainWindow, "Error", traceback.format_exc())

    def do_evaluation(self):
        try:
            if self.lm is None:
                QMessageBox.warning(self.MainWindow, "Warning", "You need to train or load a model first!")
                return

            coeff_df = pd.DataFrame(self.lm.coef_, self.X.columns, columns=['Coefficient'])
            self.textCoef.setPlainText(str(coeff_df))

            predictions = self.lm.predict(self.X_test)

            # Hiển thị dữ liệu mẫu trong bảng
            self.tableResults.setRowCount(len(self.X_test))
            self.tableResults.setColumnCount(7)
            headers = ['Avg. Area Income', 'Avg. Area House Age',
                       'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms',
                       'Area Population', 'Original Price', 'Prediction Price']
            self.tableResults.setHorizontalHeaderLabels(headers)

            y_test_array = np.asarray(self.y_test)
            for i in range(len(self.X_test)):
                values = [
                    self.X_test.iloc[i, 0],
                    self.X_test.iloc[i, 1],
                    self.X_test.iloc[i, 2],
                    self.X_test.iloc[i, 3],
                    self.X_test.iloc[i, 4],
                    y_test_array[i],
                    predictions[i]
                ]
                for j, val in enumerate(values):
                    self.tableResults.setItem(i, j, QTableWidgetItem(str(val)))

            mae = metrics.mean_absolute_error(self.y_test, predictions)
            mse = metrics.mean_squared_error(self.y_test, predictions)
            rmse = np.sqrt(mse)

            self.lineMAE.setText(f"{mae:.2f}")
            self.lineMSE.setText(f"{mse:.2f}")
            self.lineRMSE.setText(f"{rmse:.2f}")

            QMessageBox.information(self.MainWindow, "Info", "Evaluation completed successfully.")
        except Exception:
            QMessageBox.critical(self.MainWindow, "Error", traceback.format_exc())

    # ======== SAVE MODEL (có hỏi xác nhận + tự động đặt tên) ========
    def do_save_model(self):
        try:
            if self.lm is None:
                QMessageBox.warning(self.MainWindow, "Warning", "No trained model to save.")
                return

            confirm = QMessageBox.question(
                self.MainWindow,
                "Confirm Save",
                "Are you sure you want to save this model?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if confirm != QMessageBox.StandardButton.Yes:
                QMessageBox.information(self.MainWindow, "Cancelled", "Save model cancelled.")
                return

            # Tạo tên file có timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"model_{timestamp}.zip"
            with open(filename, "wb") as f:
                pickle.dump(self.lm, f)

            QMessageBox.information(self.MainWindow, "Info", f"Model saved as {filename}")
            self.refresh_model_list()

        except Exception:
            QMessageBox.critical(self.MainWindow, "Error", traceback.format_exc())

    # ======== LOAD MODEL (hiển thị danh sách và tự load khi chọn) ========
    def refresh_model_list(self):
        """Cập nhật danh sách model trong combobox"""
        self.comboModels.clear()
        models = [f for f in os.listdir() if f.endswith(".zip")]
        if not models:
            models = ["No model available"]
        self.comboModels.addItems(models)

    def do_load_model(self):
        """Nạp model được chọn trong combobox"""
        selected_model = self.comboModels.currentText()
        if selected_model == "No model available":
            QMessageBox.warning(self.MainWindow, "Warning", "No model files found.")
            return

        try:
            with open(selected_model, "rb") as f:
                self.lm = pickle.load(f)
            QMessageBox.information(self.MainWindow, "Info", f"Loaded model {selected_model} successfully.")
        except Exception:
            QMessageBox.critical(self.MainWindow, "Error", traceback.format_exc())

    def on_model_changed(self):
        """Tự động load khi chọn model khác"""
        selected_model = self.comboModels.currentText()
        if not selected_model.endswith(".zip") or not os.path.exists(selected_model):
            return
        try:
            with open(selected_model, "rb") as f:
                self.lm = pickle.load(f)
            self.statusbar.showMessage(f"Auto loaded model {selected_model}")
        except Exception:
            self.statusbar.showMessage("Failed to auto load model.")

    def do_prediction(self):
        try:
            if self.lm is None:
                QMessageBox.warning(self.MainWindow, "Warning", "You need to train or load a model first!")
                return

            X_new = [[
                float(self.lineIncome.text() or 0),
                float(self.lineAge.text() or 0),
                float(self.lineRooms.text() or 0),
                float(self.lineBedrooms.text() or 0),
                float(self.linePopulation.text() or 0)
            ]]
            prediction = self.lm.predict(X_new)
            self.linePrediction.setText(f"{prediction[0]:.2f}")
            QMessageBox.information(self.MainWindow, "Info", "Prediction completed.")
        except Exception:
            QMessageBox.critical(self.MainWindow, "Error", traceback.format_exc())
