import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
from matplotlib import cm


class Scan_Data:
    scaner = 'TiM 551'

    def __init__(self, file_data):
        self.file_data = file_data

    def load_data(self):
        if self.file_data.split('.')[-1] == 'csv':
            self.data = pd.read_csv(self.file_data, sep=';')
        elif self.file_data.split('.')[-1] == 'xls' or self.file_data.split('.')[-1] == 'xlsx':
            self.data = pd.read_excel(self.file_data)
        else:
            raise Exception('Nie obsługiwany format. Dopuszczalne formaty: csv, xlsx, xls')

    def clean_empty_columns(self):
        """
        Funtion of cleaning data from colums which dont contain data

        Arguments:
            data {[pandas.DataFrame]} --  tabular data structure with labeled axes
        """
        return self.data.dropna(axis=1, inplace=True)

    def check_Errors(self, name_col='ScanData.DeviceBlock.xbState.'):
        """
        Check that all values of column - contain in name "name_col" are equal False

        Arguments:
            data {[pandas.DataFrame]} --  tabular data structure with labeled axes
            name_col {[string]} --  name or string that column contain of column
        """
        bool_group = self.data.filter(like=name_col).eq(False).all()
        if not bool_group.all():
            error = []
            for i in range(len(bool_group)):
                if bool_group[i] == False:
                    error.append(bool_group.index[i].split('.')[-1])
                print(bool_group.index[i] + ' - '+ str(bool_group[i]))
            return error
        return True

    def search_scaner_data(self, col_name='.aData['):
        """[summary]

        Arguments:
            col_name {string} -- verb that we will search data column

        Returns:
            data value (distance from angle) from scanner
        """
        self.scan_data = self.data.filter(like=col_name)

    def measure_x_y(self, distance, angle):
        """
        Return set of distance on x and y axes

        Arguments:
            distance {[int:in [mm] ]} -- distance in mm from point given an angle and distance
            angle {[int : [degree] ]} -- angle in degree from point given an angle and distance
        """
        return (distance * np.cos((angle* np.pi )/ 180))//10, (distance * np.sin((angle* np.pi )/ 180))//10

    def generate_axis(self, angle):
        """
        calculate axis x and y from value of distance and angle read from scanner

        Arguments:
            angle {[degree : string]} -- Starting angle set on device
        """

        self.x_array = np.array([])
        self.y_array = np.array([])
        measure_x_y = lambda distance, start_angle: ((distance * np.cos((start_angle* np.pi )/ 180))//10, (distance * np.sin((start_angle* np.pi )/ 180))//10)
        for ia in np.arange(len(self.scan_data)):
            lista_x = []
            lista_y = []
            for i in np.arange(len(self.scan_data.iloc[ia])):
                x, y = measure_x_y(distance=self.scan_data.iloc[ia][i], start_angle=(angle+i))
                lista_x.append(x)
                lista_y.append(y)
            self.x_array = np.append(self.x_array, np.array(lista_x))
            self.y_array = np.append(self.y_array, np.array(lista_y))

    def create_dataframe_data(self):
        """
        Create pandas DataFrame wit multiindex which describe numbers of measurement and point on 2d plot
        """

        # multiiindex
        index = np.arange(len(self.scan_data))
        index2  = np.arange(len(self.scan_data.iloc[0]))
        lev = [index, index2]
        mindex = pd.MultiIndex.from_product(lev, names=['pomiar','punkt'])
        # Data Frame
        self.df = pd.DataFrame({'oś x': self.x_array, 'oś y' : self.y_array}, index=mindex)

    def generate_3D(self):
        """
        Generate 3d view
        axis Z is base on numers of measurement
        axis X is based on calcumate value from generate_axis method
        axis Y is based on calcumate value from generate_axis method
        Returns:
            3D pandas Wireframe
        """

        fig = plt.figure()
        ax= fig.add_subplot(111, projection='3d')
        x = self.df['oś x'].values.reshape(len(self.scan_data), len(self.scan_data.iloc[0])) # or x_array.reshape(len(scan_data)
        y = self.df['oś y'].values.reshape(len(self.scan_data), len(self.scan_data.iloc[0])) # or y_array.reshape(len(scan_data)
        z = self.df.index.labels[0].reshape(len(self.scan_data), len(self.scan_data.iloc[0]))
        return ax.plot_wireframe(x, y, z)

    def generate_2D(self, z_point):
        """
        Generate 2d view
        axis Z is base on numers of measurement
        axis X is based on calcumate value from generate_axis method
        axis Y is based on calcumate value from generate_axis method
        Arguments:
            angle {[degree : string]} -- Starting angle set on device
        Returns:
            2D pandas plot view
        """
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        ax1.plot(self.df['oś x'][z_point], self.df['oś y'][z_point], linestyle='dashed', marker='o',color='g', label='Styl domyślny')
        ax2.scatter(self.df['oś x'][z_point], self.df['oś y'][z_point])
        ax1.set(title='Wykres 2D pobrany z TiM551')
        ax2.set(xlabel='szerokość w cm', ylabel='pionowa odległość w cm',)