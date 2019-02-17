import sys
import os
import time
from PyQt5 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from ui.main import Ui_MainWindow
from app import Scan_Data


class mywindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(mywindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle('TiM 551 analiza jezdni')
        self.setWindowIcon(QtGui.QIcon('icon.png'))
        self.ui.QStackedWidget.setCurrentIndex(0)
        self.ui.close_PB_0.clicked.connect(self.btnCloseClicked)
        self.ui.next_PB_0.clicked.connect(self.fill_form_page_0)
        self.ui.file_TB.clicked.connect(self.select_file)

    def view_first(self):
        self.ui.QStackedWidget.setCurrentIndex(0)
        # self.showMaximized()

    def btnCloseClicked(self):
        self.close()

    def show_preview_page(self, page):
        lambda page: self.ui.QStackedWidget.setCurrentIndex(page)

    def check_file(self):
        self.angle = self.ui.angle_SB.text()

    def fill_form_page_0(self):
        self.project = self.ui.project_LE.text()
        if not self.project:
            QtWidgets.QMessageBox.about(self, "Nazwa projektu wymagana!", "Aby kontynuować uzupełnij nazwe projektu.")
            return
        self.file = self.ui.file_LE.text()
        if not self.file:
            QtWidgets.QMessageBox.about(self, "Plik wymagany!", "Aby kontynuować wybierz plik z pomiarem.")
            return
        if self.file.split('.')[-1] not in ['csv','xlsx']:
            QtWidgets.QMessageBox.about(self, "Zły format pliku!", "Plik musi mieć format .csv lub .xlsx!")
            return
        if not os.path.exists(self.file):
            QtWidgets.QMessageBox.about(self, "Plik nieistnieje!", "Wybrany plik nieistnieje.")
            return
        self.scan = Scan_Data(self.file)
        self.scan.load_data()
        if self.scan.check_empty_data_tim():
            QtWidgets.QMessageBox.about(self, "Błąd pliku!", "Wybrany plik nie zawiera danych ze skanera.")
            return
        self.scan.clean_empty_columns
        self.scan.search_scaner_data()
        self.view_page_1()

    def select_file(self):
        file, ext = QtWidgets.QFileDialog.getOpenFileName(self, 'Wybierz plik pomiarowy', "", "Pliki z danymi (*.csv *.xlsx)")
        if file:
            self.ui.file_LE.setText(file)

    def view_page_1(self):
        self.ui.progressBar.hide()
        self.ui.project_L.setText(self.project)
        self.ui.file_L.setText(self.file.split('/')[-1])
        self.ui.QStackedWidget.setCurrentIndex(1)
        self.setWindowTitle('TiM 551 analiza jezdni - ' + self.project)
        self.ui.back_PB_1.clicked.connect(self.show_previous_page)
        self.ui.close_PB_1.clicked.connect(self.btnCloseClicked)
        self.ui.angle_SB.setMaximum(225)
        self.ui.angle_SB.setMinimum(-45)
        self.ui.x_LEFT_max_SP.setMaximum(100000)
        self.ui.x_RIGHT_max_SP.setMaximum(100000)
        self.scaner_errors = self.scan.check_Errors()
        if self.scaner_errors:
            self.ui.errors_L.setText(str(len(self.scan.check_Errors())))
        elif self.scaner_errors == None:
            self.ui.errors_L.setText('Brak')
        if self.scaner_errors:
            err = '\n'.join([str('- ' + x) for x in self.scaner_errors])
            QtWidgets.QMessageBox.about(self, "Błąd urządzenia!", f"W zapisanym pliku wykryto błąd urządzenia :\n {err} .")
        self.ui.project_L.setText(self.project)
        self.ui.next_PB_1.clicked.connect(self.fill_form_page_1)

    def fill_form_page_1(self):
        try:
            self.complate = 0
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            self.ui.progressBar.setValue(self.complate)
            self.ui.progressBar.show()
            if self.scaner_errors:
                QtWidgets.QApplication.restoreOverrideCursor()
                err = '\n'.join([str('- ' + x) for x in self.scaner_errors])
                QtWidgets.QMessageBox.about(self, "Błąd urządzenia!", f"W zapisanym pliku wykryto błąd urządzenia :\n {err} .")
                self.ui.progressBar.hide()
                return
            self.complate = 10
            self.ui.progressBar.setValue(self.complate)
            self.start_angle = self.ui.angle_SB.value()
            self.complate = 23
            self.ui.progressBar.setValue(self.complate)
            self.scan.create_dataframe(self.start_angle)
            self.complate = 40
            self.ui.progressBar.setValue(self.complate)
            self.x_l = self.ui.x_LEFT_max_SP.value()
            self.x_r = self.ui.x_RIGHT_max_SP.value()
            self.complate = 66
            self.ui.progressBar.setValue(self.complate)
            self.scan.select_range_measure(self.x_l, self.x_r)
            self.complate = 69
            self.ui.progressBar.setValue(self.complate)
            if self.ui.direct_1_RB.isChecked() == True:
                self.scan.reg_view(1)
            elif self.ui.direct_2_RB.isChecked()== True:
                self.scan.reg_view(2)
            else:
                QtWidgets.QApplication.restoreOverrideCursor()
                QtWidgets.QMessageBox.about(self, "Brak zaznaczenia!", "Aby kontynować musisz zaznaczyć charakter przekroju poprzecznego jezdni.")
                self.ui.progressBar.hide()
                return
            self.complate = 75
            self.ui.progressBar.setValue(self.complate)
            # if self.scan.df_reg['area'].min() > 15000:
            self.scan.clean_antiregression_data(10000)
            # elif self.scan.df_reg['area'].min() < 15000:
            #     # stupid selection
            #     val = self.scan.df_reg['area'][self.scan.df_reg['area'] < self.scan.df_reg['area'][self.scan.df_reg['area']<self.scan.df_reg['area'].median()].mean()].mean()
                # self.scan.clean_antiregression_data(val)
            self.complate = 82
            self.ui.progressBar.setValue(self.complate)
            self.scan.correct_regression()
            self.complate = 87
            self.ui.progressBar.setValue(self.complate)
            self.plot_data_p2()
            while self.complate < 100:
                time.sleep(0.05)
                self.complate += 1
                self.ui.progressBar.setValue(self.complate)
            self.ui.progressBar.hide()
            QtWidgets.QApplication.restoreOverrideCursor()
            self.view_page_2()
        except:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.about(self, "Nieznany błąd!", "Wystąpił nieznany błąd, sprawdź czy: \n- wgrałeś odpowiedni plik,\n- wprowadziłeś poprawne dane: kąt początkowy, zakres szerokości.")
            self.ui.progressBar.hide()
            return

    def view_page_2(self):
        self.ui.QStackedWidget.setCurrentIndex(2)
        self.ui.back_PB_2.clicked.connect(self.show_previous_page)
        self.ui.close_PB_2.clicked.connect(self.btnCloseClicked)
        self.ui.next_PB_2.clicked.connect(self.view_page_3)
        self.ui.save_PB_2.clicked.connect(self.save_last_df)

    def save_last_df(self):
        # options = QtWidgets.QFileDialog.Options()
        # options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, ext = QtWidgets.QFileDialog.getSaveFileName(self,"QtWidgets.QFileDialog.getSaveFileName()","","Pliki z danymi (*.xlsx)")
        if fileName:
            f = fileName
            if f.split('.')[-1] != 'xlsx':
                f = fileName + '.xlsx'
            self.scan.save_data_excel(f)
            if os.path.isfile(f):
                QtWidgets.QMessageBox.about(self, "Zapisano pomyślnie !", "Gratuluje, pomyślnie zapisano dane.")

    def plot_data_p2(self):
        self.ui.sum_a_L.setText(str(round(self.scan.df_reg.sum()['area']*0.01, 2)) + ' cm\u00b2')
        self.ui.max_a_L.setText(str(round(self.scan.df_reg.max()['area']*0.01, 2)) + ' cm\u00b2')
        self.ui.max_a_L_des.setText('Punkt pomiarowy '+ str(self.scan.df_reg['area'].idxmax())) 
        self.ui.min_a_L.setText(str(round(self.scan.df_reg.min()['area']*0.01, 2)) + ' cm\u00b2')
        self.ui.min_a_L_des.setText('Punkt pomiarowy '+ str(self.scan.df_reg['area'].idxmin()))
        if 'coef' in self.scan.df_reg.columns:
            self.ui.mean_p_L.setText(str(round(self.scan.df_reg['coef'].mean(),2)) + ' %')
        else:
            self.ui.mean_p_L.setText(str(round(self.scan.df_reg['coef_1'].mean() ,2)) + ' % ; ' + str(round(self.scan.df_reg['coef_2'].mean(),2)) + ' % ')
        x, y = self.scan.df_reg.index.values, self.scan.df_reg['area'].values
        self.ui.plotWidget.canvas.ax.clear()
        self.ui.plotWidget.canvas.ax.plot(x, y)
        self.ui.plotWidget.canvas.ax.set(title='Powierzchnia nierównosci w punktach pomiarowych.', xlabel='Punkt pomiarowy', ylabel='Pole powierzchni nierówności [mm\u00b2]')
        self.ui.plotWidget.canvas.draw()

    def plot_data_p3(self, point=0):
        x = self.scan.df_reg_data.loc[point]['oś x']
        y = self.scan.df_reg_data.loc[point]['oś y']
        reg_y = self.scan.df_reg_data.loc[point]['reg y']
        self.ui.plotWidget_2D.canvas.ax.clear()
        self.ui.plotWidget_2D.canvas.ax.plot(x, y, 'bo--', linewidth=0.5, markersize=0.5)
        self.ui.plotWidget_2D.canvas.ax.plot(x, reg_y, 'r-', linewidth=1)
        self.ui.plotWidget_2D.canvas.ax.fill_between(x, y, reg_y,  color=(0, 0, 0, 0.1),hatch='|')
        self.ui.plotWidget_2D.canvas.ax.legend(['punkty pobrane z skanera','regresja liniowa'])
        x_min = round(max(self.scan.df_reg_data['oś x'].min(), (-1)*self.x_l))
        x_max = round(min(self.scan.df_reg_data['oś x'].max(), self.x_r))
        self.ui.plotWidget_2D.canvas.ax.set_xlim(x_min, x_max)
        y_min = round(min(self.scan.df_reg_data['oś y'].min(), self.scan.df_reg_data['reg y'].min()))
        y_max = round(max(self.scan.df_reg_data['oś y'].max(), self.scan.df_reg_data['reg y'].max()))
        self.ui.plotWidget_2D.canvas.ax.set_ylim(y_min, y_max)
        self.ui.plotWidget_2D.canvas.ax.grid(True)
        self.ui.plotWidget_2D.canvas.ax.set(title=('Wykres 2D pobrany z TiM551 - punkt ' + str(point)), xlabel='odległość pozioma [mm]', ylabel='pionowa odległość [mm]')
        self.ui.plotWidget_2D.canvas.draw()
        self.ui.sum_a3_L.setText(str(round(self.scan.df_reg_data.loc[point]['area'].sum()*0.01,2)) + ' cm\u00b2')
        self.ui.max_a3_L.setText(str(round(self.scan.df_reg_data.loc[point]['z'].abs().max(),2)) + ' mm')
        if 'coef' in self.scan.df_reg.columns:
            self.ui.p_a3_L.setText(str(self.scan.df_reg.loc[point]['coef']) + ' %')
        else:
            self.ui.p_a3_L.setText(str(self.scan.df_reg.loc[point]['coef_1']) + ' % ; ' + str(self.scan.df_reg.loc[point]['coef_2']) +' % ')

    def view_page_3(self):
        self.ui.QStackedWidget.setCurrentIndex(3)
        self.ui.back_PB_3.clicked.connect(self.show_previous_page)
        self.ui.close_PB_3.clicked.connect(self.btnCloseClicked)
        self.plot_data_p3()
        self.ui.point_HS.setMaximum(len(self.scan.df_reg)-1)
        self.ui.point_SB.setMaximum(len(self.scan.df_reg)-1)
        self.ui.point_HS.valueChanged.connect(self.hs_value_change)
        self.ui.point_SB.valueChanged.connect(self.sb_value_change)

    def show_previous_page(self):
        self.ui.QStackedWidget.setCurrentIndex(self.ui.QStackedWidget.currentIndex()-1)

    def hs_value_change(self):
        self.ui.point_SB.setValue(self.ui.point_HS.value())
        self.plot_data_p3(self.ui.point_HS.value())

    def sb_value_change(self):
        self.ui.point_HS.setValue(self.ui.point_SB.value())
        self.plot_data_p3(self.ui.point_SB.value())

    # po nacisniecie f12 zamyka sie okno
    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_F12:
            self.close()


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    application = mywindow()
    application.show()
    sys.exit(app.exec())

