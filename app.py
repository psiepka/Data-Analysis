import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
from matplotlib import cm
from sklearn import linear_model


class Scan_Data:
    scaner = 'TiM 551'

    def __init__(self, file_data):
        self.file_data = file_data

    def load_data(self):
        if self.file_data.split('.')[-1] == 'csv':
            self.data = pd.read_csv(self.file_data, sep=';', engine='python')
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

    
#     def generate_axis(self, angle):
#         """
#         calculate axis x and y from value of distance and angle read from scanner

#         Arguments:
#             angle {[degree : string]} -- Starting angle set on device
        
#         lambda measure_x_y
#             Return set of distance on x and y axes

#         Arguments:
#             distance {[int:in [mm] ]} -- distance in mm from point given an angle and distance
#             angle {[int : [degree] ]} -- angle in degree from point given an angle and distance
#         """

#         self.x_array = np.array([])
#         self.y_array = np.array([])  
#         measure_x_y = lambda distance, start_angle: ((distance * np.cos((start_angle* np.pi )/ 180)), (distance * np.sin((start_angle* np.pi )/ 180)))
#         for ia in np.arange(len(self.scan_data)):
#             lista_x = []
#             lista_y = []
#             for i in np.arange(len(self.scan_data.iloc[ia])):
#                 x, y = measure_x_y(distance=self.scan_data.iloc[ia][i], start_angle=(angle+i))
#                 lista_x.append(x)
#                 lista_y.append(y)
#             self.x_array = np.append(self.x_array, np.array(lista_x))
#             self.y_array = np.append(self.y_array, np.array(lista_y))

#     def create_dataframe_data(self, speed=None, Hz=15):
#         """
#         Create pandas DataFrame wit multiindex which describe numbers of measurement and point on 2d plot
        
#         Z axis describe on what cm since start measure plot is generate,
#         scaner TiM 551 is working with 15 Hz that mean in one sec it take 15 measure point,
#         accorgin to speed we know how many cm is measure in one sec
        
#         args: 
#             speed -  measurement in km/h 
#             Hz - how many measurment device can take in one secound, (default : 15 Hz - 15 point on sec)
#         """
#         # multiiindex
#         if speed:
#             func = lambda x, t: np.round(x*t)
#             vfunc = np.vectorize(func)
#             self.interval = ((((speed*100000)/3600))/Hz)
#             index = vfunc(np.arange(len(self.scan_data)), self.interval)
#         else:
#             index = np.arange(len(self.scan_data))
#             self.interval = 1
#         index2  = np.arange(len(self.scan_data.iloc[0]))
#         lev = [index, index2]
#         mindex = pd.MultiIndex.from_product(lev, names=['pomiar','punkt'])
#         # Data Frame
#         self.df = pd.DataFrame({'oś x': self.x_array, 'oś y' : self.y_array}, index=mindex)
#         # apply mean of measure point for the same value of index  
#         if self.interval < 1:
#             self.df = self.df.groupby([self.df.index.get_level_values('pomiar'), self.df.index.get_level_values('punkt')]).mean()

    def create_dataframe(self, start_angle, speed=None, Hz=15):
        """
        calculate axis x and y from value of distance and angle read from scanner

        Arguments:
            angle {[degree : string]} -- Starting angle set on device
        
        lambda measure_x_y
            Return set of distance on x and y axes

        Arguments:
            distance {[int:in [mm] ]} -- distance in mm from point given an angle and distance
            angle {[int : [degree] ]} -- angle in degree from point given an angle and distances
            
        Create pandas DataFrame wit multiindex which describe numbers of measurement and point on 2d plot
        
        Z axis describe on what cm since start measure plot is generate,
        scaner TiM 551 is working with 15 Hz that mean in one sec it take 15 measure point,
        accorgin to speed we know how many cm is measure in one sec
        
        args: 
            speed -  measurement in km/h 
            Hz - how many measurment device can take in one secound, (default : 15 Hz - 15 point on sec)
    
        
        """
        if speed:
            func = lambda x, t: int(x*t)
            vfunc = np.vectorize(func)
            interval = ((((speed*100000)/3600))/Hz)
            index = vfunc(np.arange(len(self.scan_data)), interval)
            self.scan_data.index = index
        index_angle = range(start_angle, start_angle+len(self.scan_data.columns))
        data = self.scan_data.rename({a:x for x, a in zip(index_angle, self.scan_data.columns)}, axis='columns')
        d1 = pd.DataFrame(data.stack())
        d1['x'], d1['y']= d1.apply(lambda x : np.round((x*np.cos((x.name[1]*np.pi)/180)), 2), axis=1), d1.apply(lambda x : np.round((x*np.sin((x.name[1]*np.pi)/180)), 2), axis=1)
        d1.columns = ['odległość','oś x','oś y']
        d1.index.names = ['pomiar', 'kąt pomiaru']
        if interval < 1:
            d1 = d1.groupby(['pomiar', 'kąt pomiaru']).mean()
        self.df = d1


    def generate_3D(self):
        """
        Generate 3d view
        axis Z is base on measurement point from start of measure in cm 
        
        axis X is based on calcumate value from generate_axis method in mm
        axis Y is based on calcumate value from generate_axis method in mm
        Returns:
            3D pandas Wireframe
        """
        fig = plt.figure()
        ax= fig.add_subplot(111, projection='3d')
        x = self.df['oś x'].values.reshape(len(self.df.index.levels[0]), len(self.scan_data.iloc[0])) # or x_array.reshape(len(scan_data)
        y = self.df['oś y'].values.reshape(len(self.df.index.levels[0]), len(self.scan_data.iloc[0])) # or y_array.reshape(len(scan_data)
        z = np.array([self.df.index.levels[0][x] for x in self.df.index.labels[0]]).reshape(len(self.df.index.levels[0]), len(self.scan_data.iloc[0]))
        return ax.plot_wireframe(x, z, y)

    def generate_2D(self, z_point, condition=None):
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
        ax1.set(title='Wykres 2D pobrany z TiM551')
        ax2.set(xlabel='szerokość w cm', ylabel='pionowa odległość w mm',)
        if condition:
            ax1.plot(self.df['oś x'][z_point][condition], self.df['oś y'][z_point][condition], linestyle='dashed', color='g', label='Styl domyślny')
            ax2.scatter(self.df['oś x'][z_point], self.df['oś y'][z_point])
        else:
            ax1.plot(self.df['oś x'][z_point], self.df['oś y'][z_point], linestyle='dashed', color='g', label='Styl domyślny')
            ax2.scatter(self.df['oś x'][z_point], self.df['oś y'][z_point])
        

    def create_regression(self, z_point, directional):
        """
        directional : 1 or 2 - cross-section of road direction
        """
        if directional == 1:
            x = self.df['oś x'][z_point].values.reshape(-1, 1) 
            y = self.df['oś y'][z_point].values.reshape(-1, 1)
            reg = linear_model.LinearRegression()
            reg.fit(x, y)
            pred = reg.predict(x)
            
            
            self.slope = reg.coef_
            self.reg_c = reg.intercept_
            self.regression = reg
            self.pred = pred
            
            
            fig = plt.figure()
            ax1 = fig.add_subplot(3, 1, 1)
            ax2 = fig.add_subplot(3, 1, 2)
            ax3 = fig.add_subplot(3, 1, 3)
            ax1.set(title='Wykres 2D pobrany z TiM551 z wkresloną regresją liniową')
            ax1.set(xlabel='szerokość w cm', ylabel='pionowa odległość w mm',)
            ax1.scatter(x, y)
            ax1.plot(x, pred, 'b-', linewidth=2,)
            
            
            d_5 = self.df.loc[5]
            d_5['reg y']=self.pred
            d_5['z'] = d_5['oś y'] - d_5['reg y']
            
            x2 = d_5['oś x'].shift(1)
            x1 = d_5['oś x']
            z2 = d_5['z'].shift(1)
            z1 = d_5['z']

            xi = x1-z1*(x1-x2)/(z1-z2)
            dxi = -z1*(x1-x2)/(z1-z2)
            dx = x2-x1

            area_pos = 0.5*dx*abs(z1+z2)
            area_neg = 0.5*dxi*abs(z1) +0.5*(dx-dxi)*abs(z2)

            d_5['area'] = np.where(d_5['z']*d_5['z'].shift(1)<0, area_neg, area_pos)
            asd = 'self.data_'+str(z_point)
            self.asd = d_5
            self.d = d_5
            
            x2 = self.df['oś x'][z_point][self.df.loc[z_point]['oś y'].diff().abs()<self.df.loc[z_point]['oś y'].std()].values.reshape(-1, 1) 
            y2 = self.df['oś y'][z_point][self.df.loc[z_point]['oś y'].diff().abs()<self.df.loc[z_point]['oś y'].std()].values.reshape(-1, 1)
            reg2 = linear_model.LinearRegression()
            reg2.fit(x2, y2)
            pred2 = reg2.predict(x2)
            
            
            
            self.slope_opt = reg2.coef_
            self.reg_c_opt = reg2.intercept_
            
            
            ax2.scatter(x2, y2)
            ax2.plot(x2, pred2, 'r-', linewidth=2,)
            ax3.scatter(x,y)
            ax3.plot(x, pred, 'b-', x2, pred2, 'r--')
            
            
        elif directional == 2:
            x1 = self.df['oś x'][z_point][self.df['oś x'][z_point]>0].values.reshape(-1, 1) 
            y1 = self.df['oś y'][z_point][self.df['oś x'][z_point]>0].values.reshape(-1, 1)
            x2 = self.df['oś x'][z_point][self.df['oś x'][z_point]<0].values.reshape(-1, 1) 
            y2 = self.df['oś y'][z_point][self.df['oś x'][z_point]<0].values.reshape(-1, 1)
            reg1 = linear_model.LinearRegression()
            reg1.fit(x1, y1)
            pred1 = reg1.predict(x1)
            reg2 = linear_model.LinearRegression()
            reg2.fit(x2, y2)
            pred2 = reg2.predict(x2)
            self.slope = [reg1.coef_, reg2.coef_]
            self.reg_c = [reg1.intercept_, reg2.intercept_]
            fig = plt.figure()
            ax1 = fig.add_subplot(2, 1, 1)
            ax2 = fig.add_subplot(2, 1, 2)
            ax1.set(title='Wykres 2D pobrany z TiM551 z wkresloną regresją liniową')
            ax2.set(xlabel='szerokość w cm', ylabel='pionowa odległość w mm',)
            ax1.scatter(x1, y1)
            ax1.plot(x1, pred1, 'r--', x2, pred2, 'b-')
            ax2.scatter(x2, y2)
            ax2.plot(x2, pred2, 'b-', x1, pred1, 'r--')
        else:
            raise ValueError('Directional can have valeue 1 or 2.')