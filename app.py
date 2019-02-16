import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
from matplotlib import cm
from sklearn import linear_model


class Scan_Data:
    scaner = 'TiM 551'
    Hz = 15

    def __init__(self, file_data):
        self.file_data = file_data

    @classmethod
    def set_Hz_device(cls, value):
        """changing device parametr
        
        Arguments:
            value [int or float] -- frequency of device in [Hz]
        """
        cls.Hz = value

    @classmethod
    def set_scanner_device(cls, name):
        """changing device parametr
        
        Arguments:
            name [str] --name of scanner
        """
        cls.scaner = name

    def load_data(self):
        """ load data from file
        
        Raises:
            Exception -- allowed format csv and exels (xls, xlsx)
        """

        if self.file_data.split('.')[-1] == 'csv':
            self.data = pd.read_csv(self.file_data, sep=';', engine='python')
        elif self.file_data.split('.')[-1] == 'xls' or self.file_data.split('.')[-1] == 'xlsx':
            self.data = pd.read_excel(self.file_data)
        else:
            raise Exception('Nie obsługiwany format. Dopuszczalne formaty: csv, xlsx, xls')

    def check_empty_data_tim(self):
        """check is file contain data what we need to calculate
        
        Returns:
            False if everything is fine
        """

        if (self.data.filter(like='.aData[')).empty and self.data.filter(like='ScanData.DeviceBlock.xbState.').empty:
            return True
        return False

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
    
    def search_scaner_data(self, col_name='.aData['):
        """[summary]

        Arguments:
            col_name {string} -- verb that we will search data column

        Returns:
            data value (distance from angle) from scanner
        """
        self.scan_data = self.data.filter(like=col_name)
        self.scan_data.replace(0,np.nan, inplace=True)


    def create_dataframe(self, start_angle, speed=None):
        """
        - calculate axis x and y from value of distance and angle read from scanner
        - cleaning error measure points 

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
            Scan_Data.Hz - how many measurment device can take in one secound, (default : 15 Hz - 15 point on sec) - to change value "Scan_Data.set_Hz_device(value)
        """
        if speed:
            func = lambda x, t: int(x*t)
            vfunc = np.vectorize(func)
            interval = ((((speed*100000)/3600))/Scan_Data.Hz)
            index = vfunc(np.arange(len(self.scan_data)), interval)
            self.scan_data.index = index
        index_angle = range(start_angle, start_angle+len(self.scan_data.columns))
        data = self.scan_data.rename({a:x for x, a in zip(index_angle, self.scan_data.columns)}, axis='columns')
        d1 = pd.DataFrame(data.stack())
        d1['x'], d1['y']= d1.apply(lambda x : np.round((x*np.cos((x.name[1]*np.pi)/180)), 2), axis=1), d1.apply(lambda x : np.round((x*np.sin((x.name[1]*np.pi)/180))*(-1), 2), axis=1)
        d1.columns = ['odległość','oś x','oś y']
        d1.index.names = ['pomiar', 'kąt pomiaru']
        if speed:
            if interval < 1:
                d1 = d1.groupby(['pomiar', 'kąt pomiaru']).mean()
        self.df = d1
        self.df[self.df['odległość'].diff().abs()> 1000] = np.nan
        self.df[self.df['odległość'].abs() < 10] = np.nan


    def select_range_measure(self, x_left, x_right):
        """change range of calcualting data
        
        Arguments:
            x_left {[int]} -- range from device in left direction
            x_right {[int]} -- range from device in right direction
        """

        self.df = self.df[(self.df['oś x']>(-x_left)) & (self.df['oś x']<x_right)]

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
            ax1.plot(self.df['oś x'][z_point], self.df['oś y'][z_point], linestyle='dashed', color='g')
            ax2.scatter(self.df_reg_data['oś x'][z_point], self.df_reg_data['oś y'][z_point])
            ax2.plot(self.df_reg_data['oś x'][z_point], self.df_reg_data['reg y'][z_point], 'b-')


    @staticmethod
    def data_create_regression(data, directional):
        """
        directional : 1 or 2 - cross-section of road direction
        """
        if directional == 1:
            x = data['oś x'].values.reshape(-1, 1) 
            y = data['oś y'].values.reshape(-1, 1)
            reg = linear_model.LinearRegression()
            reg.fit(x, y)
            pred = reg.predict(x)
            d_5 = data
            d_5['reg y']=pred
            d_5['z'] = d_5['oś y'] - d_5['reg y']
            d_5['coef'] = round(float(reg.coef_*100),2)
        elif directional == 2:
            x1 = data['oś x'][data['oś x']>=0].values.reshape(-1, 1)
            y1 = data['oś y'][data['oś x']>=0].values.reshape(-1, 1)
            x2 = data['oś x'][data['oś x']<=0].values.reshape(-1, 1)
            y2 = data['oś y'][data['oś x']<=0].values.reshape(-1, 1)
            reg1 = linear_model.LinearRegression()
            reg1.fit(x1, y1)
            pred1 = reg1.predict(x1)
            reg2 = linear_model.LinearRegression()
            reg2.fit(x2, y2)
            pred2 = reg2.predict(x2)
            pred1[-1] = (pred1[-1]+pred2[0])/2
            reg_y = np.round(np.append(pred1, pred2[1:]),2)
            d_5 = data
            d_5['reg y'] = reg_y
            d_5['z'] = d_5['oś y'] - d_5['reg y']
            d_5['coef'] = np.nan
            d_5['coef'][data['oś x']>=0] = round(float(reg1.coef_*100),2)
            d_5['coef'][data['oś x']<0] = round(float(reg2.coef_*100),2)
            # # optimization measure point -- without sucess
            # x1_op = data['oś x'][(data['oś y'].diff().abs()<data['oś y'].std()) & (data['oś x']>=0)].values.reshape(-1, 1) 
            # y1_op = data['oś y'][(data['oś y'].diff().abs()<data['oś y'].std()) & (data['oś x']>=0)].values.reshape(-1, 1)
            # reg1_op = linear_model.LinearRegression()
            # reg1_op.fit(x1_op, y1_op)
            # pred1_op = reg1_op.predict(x1)
            # x2_op = data['oś x'][(data['oś y'].diff().abs()<data['oś y'].std()) & (data['oś x']<=0)].values.reshape(-1, 1) 
            # y2_op = data['oś y'][(data['oś y'].diff().abs()<data['oś y'].std()) & (data['oś x']<=0)].values.reshape(-1, 1)
            # reg2_op = linear_model.LinearRegression()
            # reg2_op.fit(x2_op, y2_op)
            # pred2_op = reg2_op.predict(x2)
            # pred1_op[-1] = (pred1_op[-1]+pred2_op[0])/2
            # reg_y_op = np.round(np.append(pred1_op, pred2_op[1:]),2)
            # data['reg y_op'] = reg_y_op

        # Calculate area between plots
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
        return d_5

    @staticmethod
    def correct_area(d_5):
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
        return d_5

    def reg_view(self, directional):
        """
        directional : 1 or 2 - cross-section of road direction
        """
        if directional != 1 and directional != 2:
            raise ValueError('Directional must have valeue 1 or 2.')
        self.df_reg_data = self.df.groupby([self.df.index.get_level_values('pomiar')]).apply(self.data_create_regression, directional)
        if directional == 2:
            self.df_reg = self.df_reg_data.groupby([self.df.index.get_level_values('pomiar')]).agg({'area':np.sum,'coef':lambda x : [x.iloc[-1],(-1)*x.iloc[0]]})
            self.df_reg['coef_1'], self.df_reg['coef_2'] =zip(*self.df_reg['coef'].apply(lambda x : (x[0], x[1])))
            del self.df_reg['coef']
        elif directional ==1:
            self.df_reg = self.df_reg_data.groupby([self.df.index.get_level_values('pomiar')]).agg({'area':np.sum,'coef':lambda x : x.iloc[0]})

    def clean_antiregression_data(self, area_allowed):
        """
        Delete regression where area of inequalities road are greater than allowed area [mm ^2]
        Arguments:
            area_allowed {[int} --  allowed inequalities area [mm ^2]
        """
        indexs = list(self.df_reg[self.df_reg['area']>area_allowed].index)
        if len(indexs) > 0:
            for x in indexs:
                self.df_reg_data['reg y'].loc[x] = np.nan

    def correct_regression(self):
        """
        Interpolate beetwen point that we cant calculate with regression and save in our result Dataframes
        """

        self.df_reg_data['reg y'] = self.df_reg_data['reg y'].groupby([self.df_reg_data.index.get_level_values('kąt pomiaru')]).apply(lambda x: x.interpolate())
        self.df_reg_data = self.df_reg_data.groupby([self.df_reg_data.index.get_level_values('pomiar')]).apply(self.correct_area)
        self.df_reg['area'] = self.df_reg_data['area'].groupby([self.df.index.get_level_values('pomiar')]).apply(np.sum)