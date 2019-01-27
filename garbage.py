
# File localization
file = 'sciana1.csv'

# read file data
if file.split('.')[-1] == 'csv':
    data = pd.read_csv(file, sep=';')
elif file.split('.')[-1] == 'xls' or file.split('.')[-1] == 'xlsx':
    data = pd.read_excel(file)

data = pd.read_csv('sciana1.csv', sep=';')

clean_data = data.dropna(axis=1)

# scan_data

scan_data = clean_data.filter(like='.aData[')


def clean_data_col(data):
    """
    Funtion of cleaning data from colums which dont contain data

    Arguments:
        data {[pandas.DataFrame]} --  tabular data structure with labeled axes
    """
    return data.dropna(axis=1)

## TESTUJ GÓRA



def measure_x_y(distance, angle):
    """
    Return set of distance on x and y axes

    Arguments:
        distance {[int:in [mm] ]} -- distance in mm from point given an angle and distance
        angle {[int : [degree] ]} -- angle in degree from point given an angle and distance
    """
    return (distance * np.cos((angle* np.pi )/ 180))//10, (distance * np.sin((angle* np.pi )/ 180))//10




data = clean_data_col(data)
print(check_False(data, 'ScanData.DeviceBlock.xbState.'))

#opcja1
d_a = {}
for ia in np.arange(len(scan_data)):
    d_a[ia] = {'Oś x':[], 'Oś y':[]}
    for i in np.arange(len(scan_data.iloc[ia])):
        x, y = measure_x_y(scan_data.iloc[ia][i], (75+i))
        d_a[ia]['Oś x'].append(x)
        d_a[ia]['Oś y'].append(y)

df1 = pd.DataFrame(d_a)


#opcja2 - moja opcja
d_a = {'Oś x': {}, 'Oś y':{}}
for ia in np.arange(len(scan_data)):
    d_a['Oś x'].update({ia:[]})
    d_a['Oś y'].update({ia:[]})
    for i in np.arange(len(scan_data.iloc[ia])):
        x, y = measure_x_y(scan_data.iloc[ia][i], (75+i))
        d_a['Oś x'][ia].append(x)
        d_a['Oś y'][ia].append(y)

df2 = pd.DataFrame(d_a)



ndar = None
for i in df2.index: # opcja 1 .columns
    if ndar is None:
        ndar = np.full((1,len(df2['Oś x'][0])),i)
    else:
        ndar = np.append(ndar, np.full((1,len(df2['Oś x'][0])),i), axis=0)

ndar

datax_array = None
datay_array = None
for ia in np.arange(len(scan_data)):
    lista_x = []
    lista_y = []
    for i in np.arange(len(scan_data.iloc[ia])):
        x, y = measure_x_y(scan_data.iloc[ia][i], (75+i))
        lista_x.append(x)
        lista_y.append(y)
    if datax_array is None:
        datax_array = np.array(lista_x)
        datay_array = np.array(lista_y)
    else:
        datax_array = np.vstack((datax_array, np.array(lista_x)))
        datay_array = np.vstack((datay_array, np.array(lista_y)))


df = pd.DataFrame(d_a)



# creating dataframe with multiindex


#data of axis
x_array = np.array([])
y_array = np.array([])
for ia in np.arange(len(scan_data)):
    lista_x = []
    lista_y = []
    for i in np.arange(len(scan_data.iloc[ia])):
        x, y = measure_x_y(scan_data.iloc[ia][i], (75+i))
        lista_x.append(x)
        lista_y.append(y)
    x_array = np.append(x_array, np.array(lista_x))
    y_array = np.append(y_array, np.array(lista_y))


# multiiindex
index = np.arange(len(scan_data))
index2  = np.arange(len(scan_data.iloc[0]))

lev = [index, index2]

mindex = pd.MultiIndex.from_product(lev, names=['pomiar','punkt'])


# Data Frame

DF = pd.DataFrame({'oś x':x_array,'oś y':y_array}, index=mindex)

# draw a datawife

fig = plt.figure()
ax= fig.add_subplot(111, projection='3d')
ax.plot_wireframe(DF['x'].values.reshape(len(scan_data), len(scan_data.iloc[0])), DF['y'].values.reshape(len(scan_data), len(scan_data.iloc[0])), DF.index.labels[0].reshape(len(scan_data), len(scan_data.iloc[0])))