import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import os
import webcolors
from osgeo import gdal

#from pycwr.io import read_auto
#from pycwr.draw.RadarPlot import plot_xy,add_rings
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
import matplotlib.colors as colors
import imp
import math

#import pyart
#from pyart.core.transforms import antenna_to_cartesian 
#from pyart.core import cartesian_to_geographic
import copy

#constant
R_earth = 6371.393

#获取图例的像素位置区域范围
left_x = 955
upper_y = 665
right_x = 973
lower_y = 858

#右下角图例显示的13个色块所表示颜色的编码，可利用MacOS上一colorSLurp的取色器取色
COLORS = [
    '#AD90F0',
    '#9600B4',
    '#FF00F0',
    '#C00001',
    '#D60100',
    '#FF0200',
    '#FF9000',
    '#E7C000',
    '#FEFF00',
    '#009000',
    '#00D800',
    '#00ECEC',
    '#01A0F6'
]

#把颜色的hex码转换为RGB整数值
colors_arrays = np.array([list(webcolors.hex_to_rgb(c)) for c in COLORS])
#print(colors_arrays)
cm_user=np.array(colors_arrays[::-1])/255.0
icmap=colors.ListedColormap(cm_user,name='my_color') 
plt.register_cmap(cmap=icmap)

class Dataset:
    def __init__(self, in_file):
        self.in_file = in_file #Tiff_file
        dataset = gdal.Open(self.in_file)
        self.XSize = dataset.RasterXSize
        self.YSize = dataset.RasterYSize
        self.GeoTransform = dataset.GetGeoTransform()
        self.GetProjection = dataset.GetProjection()
        #self.band1 = dataset.GetRasterBand(1)
    #def get_band(self):
    #    band1 = dataset.GetRasterBand(1)
    #    return band1
    def get_lat_lon(self):
        gtf = self.GeoTransform
        x_range = range(0, self.XSize)
        y_range = range(0, self.YSize)
        x, y = np.meshgrid(x_range, y_range)
        lon = gtf[0] + x*gtf[1] + y*gtf[2]
        lat = gtf[3] + x*gtf[4] + y*gtf[5]
        return lat, lon

def add_ring(ax, azmin,azmax,rings, color="#5B5B5B", linestyle='-', linewidth=0.6, **kwargs):
    #add_ring(ax1, 0,np.pi, [0, 40, 80, 120, 160], linestyle='-', linewidth=1)
    #print("min")
    #print(azmin,azmax)
    theta = np.linspace(azmin, azmax,200)
    
    for i in rings:
        x0 = i * np.cos(theta)
        y0 = i * np.sin(theta)
        gci = ax.plot(x0, y0, linestyle=linestyle, linewidth=linewidth, color=color, **kwargs) # circle
    for rad in np.arange(azmin, azmax+0.01, np.pi / 6.0):
        gci = ax.plot([0, rings[-1] * np.cos(rad)], \
                [0, rings[-1] * np.sin(rad)], \
                linestyle=linestyle, linewidth=linewidth, color=color, **kwargs) #line

def add_line_air(ax, heading_angle,angle_range,rings, color="#5B5B5B", linestyle='-', linewidth=0.6, **kwargs):
    # the boundary line
    azmin_r = np.deg2rad(heading_angle - 70)
    azmax_r = np.deg2rad(heading_angle + 70 )
    for rad in [azmin_r,azmax_r]:
        print(rad)
        gci = ax.plot([0, rings[-1] * np.cos(rad)], \
            [0, rings[-1] * np.sin(rad)], \
            linestyle=linestyle, linewidth=linewidth+1, color=color, **kwargs) #line

def rotate(x,y,alpha):
    #假设对图片上任意点(x,y)，绕一个坐标点(rx0,ry0)逆时针旋转a角度后的新的坐标设为(x0, y0)，有公式：
    # x0= (x - rx0)*cos(a) - (y - ry0)*sin(a) + rx0 ;
    # y0= (x - rx0)*sin(a) + (y - ry0)*cos(a) + ry0 ;
    alpha = np.deg2rad(alpha)
    x_r = x * np.cos(alpha) - y *np.sin(alpha)
    y_r = x * np.sin(alpha) + y *np.cos(alpha)
    return x_r, y_r

def get_z_from_radar(radar):
    """Input radar object, return z from radar (km, 2D)"""
    azimuth_1D = radar.azimuth['data']
    elevation_1D = radar.elevation['data']
    srange_1D = radar.range['data']
    sr_2d, az_2d = np.meshgrid(srange_1D, azimuth_1D)
    el_2d = np.meshgrid(srange_1D, elevation_1D)[1]
    xx, yy, zz = antenna_to_cartesian(sr_2d/1000.0, az_2d, el_2d)
    return zz + radar.altitude['data']


def plot_list_of_fields(radar, sweep=0, fields=['reflectivity'], vmins=[0],
                        vmaxs=[65], units=['dBZ'], cmaps=['RdYlBu_r'],
                        return_flag=False, xlim=[-150, 150], ylim=[-150, 150],
                        mask_tuple=None):
    num_fields = len(fields)
    if mask_tuple is None:
        mask_tuple = []
        for i in np.arange(num_fields):
            mask_tuple.append(None)
    nrows = (num_fields + 1) // 2
    ncols = (num_fields + 1) % 2 + 1
    fig = plt.figure(figsize=(14.0, float(nrows)*5.5))
    display = pyart.graph.RadarDisplay(radar)
    for index, field in enumerate(fields):
        ax = fig.add_subplot(nrows, 2, index+1)
        display.plot_ppi(field, sweep=sweep, vmin=vmins[index],
                         vmax=vmaxs[index],
                         colorbar_label=units[index], cmap=cmaps[index],
                         mask_tuple=mask_tuple[index])
        display.set_limits(xlim=xlim, ylim=ylim)
        
    plt.tight_layout()
    if return_flag:
        return display


class Ground2AC():
    def __init__(self, in_file, ac_parameters):
        self.in_file = in_file #Tiff_file
        self.Rmax_Air = ac_parameters['Rmax_Air']
        self.Rmin_Air = ac_parameters['Rmin_Air']
        self.heading_angle =  ac_parameters['heading_angle']
        self.lat =  ac_parameters['ll_center'][0]
        self.lon =  ac_parameters['ll_center'][1]
        self.range_angle = ac_parameters['range_angle']
        self.Bin_length = ac_parameters['Bin_length']

    def plot_range(self):
        ## STEP 1: generate a general view of the airborne radar range
        cmd = "gdal_translate -of GTiff -a_srs \"+proj=lcc +lat_1=30 +lat_2=62 +lat_0=39.0 +lon_0=110 +a=6378137 +b=6378137 +units=m +no_defs\" -a_ullr -3116700.0385205294 1693154.412068708 2017567.3399639379 -2737364.450775645 "
        src = self.in_file
        dst = "dbz.tif"
        os.system(' '.join([cmd, src, dst]))
        #os.system("gdal_translate -of GTiff -a_srs \"+proj=lcc +lat_1=30 +lat_2=62 +lat_0=39.0 +lon_0=110 +a=6378137 +b=6378137 +units=m +no_defs\" -a_ullr -3116700.0385205294 1693154.412068708 2017567.3399639379 -2737364.450775645 Z_RADA_C_BABJ_20220716015400_P_DOR_RDCP_R_ACHN.PNG dbz.tif")
        # 进一步转为等经纬度的GeoTiff文件
        os.system("gdalwarp -t_srs EPSG:4326 dbz.tif dbz_new_ll.tiff")
        ds = gdal.Open("dbz_new_ll.tiff")
        dataset = Dataset("dbz_new_ll.tiff")
        lat, lon = dataset.get_lat_lon() #每个格点的经纬度信息
        rbg_value = np.stack([ds.GetRasterBand(i).ReadAsArray() for i in (1,2,3)]) #每个格点的RGB信息

        min_max = [10, 75] 
        vmin, vmax = min_max
        dBZ_levels = np.arange(72.5, 7.5, -5) #由图右下角的colorbar

        dBZ_figure = np.empty(shape=(rbg_value.shape[1],rbg_value.shape[2]))
        for n, colors_array in enumerate(colors_arrays):
            dist = np.sum( (np.transpose(rbg_value,(1,2,0)) - colors_array)**2, axis = 2 )
            dbz_index = np.where( (dist == dist.min()) & (dist.min()< 10) )
            #print(dist.min())
            dBZ_figure[dbz_index] = dBZ_levels[n]-0.1 

        dBZ_figure = np.where(dBZ_figure == 0.0, np.nan , dBZ_figure)

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)

        plt.imshow(np.uint8(np.transpose(rbg_value,(1,2,0))))

        yy, xx = np.where( ( np.abs(lat - self.lat) < 0.04 )  & ( np.abs(lon - self.lon) < 0.04 ) )
        ax.scatter(xx[0], yy[0], s = 1, color = 'red')

        theta = np.linspace(-np.pi/2 + np.deg2rad(self.heading_angle), np.pi/2 + np.deg2rad(self.heading_angle), 200)
        x_list = []
        y_list = []

        for az in theta: 
            lllat = self.Rmax_Air *  np.cos(az)/111.0 + self.lat  #y = 80 * sin(az)
            lllon = np.rad2deg(self.Rmax_Air * np.sin(az) /(R_earth * np.cos(np.deg2rad(lllat)))) + self.lon
            y0, x0 = np.where( (np.abs(lat - lllat) < 0.04)  & (np.abs(lon - lllon) < 0.04))
            x_list.append(x0[0])
            y_list.append(y0[0])

        ax.plot(x_list, y_list, color='black', linewidth = 0.8) # circle
        ax.plot([x_list[0],x_list[-1]], [y_list[0],y_list[-1]], color='black', linewidth = 0.8) # line
        plt.savefig('%d range_show.png'%(self.Rmax_Air)) 

    def plot_ground2ac_white(self):
        radarX, radarY, dbz_ac = self.get_ac()
        print(radarY.min())

        fig = plt.figure(figsize=(12,6))

        #否则会画出很多蓝色
        dbz_ac = np.where(dbz_ac == 0.0, np.nan , dbz_ac)

        min_max = [10, 75] 
        vmin, vmax = min_max
        levels = MaxNLocator(nbins=13).tick_values(vmin, vmax)
        ticks = levels

        ax1 = fig.add_subplot(1,1,1)

        #ax1 = fig.add_axes([0.1, 0.3, 0.9, 0.5]) # [left, bottom, width, height]  ,facecolor='black'
        X1,Y1 = rotate(radarX, radarY, self.heading_angle) # rotate should be clockwise --> *-1
        pc = ax1.pcolormesh( X1/1000, Y1/1000, dbz_ac, shading='auto', vmin = vmin, vmax = vmax, cmap=icmap ) #zorder=0,  dbz_ac是跟着转的，因为某个点对应的X1，Y1和dbz_ac是固定的

        rings = np.linspace(0, self.Rmax_Air, 5)
        add_ring(ax1, 0,np.pi, rings, linestyle='-', linewidth=1)
        #ax1.text()
        #plot_xy(ax1, radarX, radarY, dbz_ac) ##画图显示
        ax1.set_title('Transformed airborne radar display with %d km range'%(self.Rmax_Air), fontsize=20)
        ax1.set_xlabel("Distance From Radar In East (km)", fontsize=10)

        plt.colorbar(mappable=pc, ax=ax1, orientation="vertical")
        plt.savefig('%d km_tr.png'%(self.Rmax_Air))

    def plot_ground2ac_black(self):
        radarX, radarY, dbz_ac = self.get_ac()

        min_max = [10, 75] 
        vmin, vmax = min_max

        fig = plt.figure(figsize=(12,6), facecolor='black')

        #否则会画出很多蓝色
        dbz_ac = np.where(dbz_ac == 0.0, np.nan , dbz_ac)
        levels = MaxNLocator(nbins=13).tick_values(vmin, vmax)
        ticks = levels

        ax1 = fig.add_subplot(111,facecolor='black')
        #ax1 = fig.add_axes([0.1, 0.3, 0.9, 0.5],facecolor='black') # [left, bottom, width, height]  ,facecolor='black'
        X1,Y1 = rotate(radarX, radarY, self.heading_angle) # rotate should be clockwise --> *-1

        hq_colors = ['Green', 'Yellow', 'Red']
        cmap_hq = colors.ListedColormap(hq_colors)
        rings = np.linspace(0, self.Rmax_Air, 5)
        print(rings)
        add_ring(ax1, 0,np.pi, rings, color='white', linestyle='-', linewidth=1.5)

        pc = ax1.pcolormesh( X1/1000, Y1/1000, dbz_ac, shading='auto', vmin = vmin, vmax = 50, cmap=cmap_hq ) #zorder=0,  dbz_ac是跟着转的，因为某个点对应的X1，Y1和dbz_ac是固定的

        plt.axis('on')
        plt.tick_params(axis='x',colors='white')
        plt.savefig('%d km_tr_black.png'%(self.Rmax_Air))
    
    def get_ac(self):
        print(self.in_file)
        self.save_dbz(self.in_file)
        cmd = "gdal_translate -of GTiff -a_srs \"+proj=lcc +lat_1=30 +lat_2=62 +lat_0=39.0 +lon_0=110 +a=6378137 +b=6378137 +units=m +no_defs\" -a_ullr -3116700.0385205294 1693154.412068708 2017567.3399639379 -2737364.450775645 "
        src = "dbz.png"
        dst = "dbz.tif"
        #os.system("gdal_translate -of GTiff -a_srs \"+proj=lcc +lat_1=30 +lat_2=62 +lat_0=39.0 +lon_0=110 +a=6378137 +b=6378137 +units=m +no_defs\" -a_ullr -3116700.0385205294 1693154.412068708 2017567.3399639379 -2737364.450775645 dbz.png dbz.tif")
        # 进一步转为等经纬度的GeoTiff文件
        os.system("gdalwarp -t_srs EPSG:4326 dbz.tif dbz_new.tiff")
        ds = gdal.Open("dbz_new.tiff")
        dataset = Dataset("dbz_new.tiff")
        lat, lon = dataset.get_lat_lon() #每个格点的经纬度信息
        rbg_value = np.stack([ds.GetRasterBand(i).ReadAsArray() for i in (1,2,3)]) #每个格点的RGB信息

        dBZ_levels = np.arange(72.5, 7.5, -5) #由图右下角的colorbar

        dBZ_figure = np.empty(shape=(rbg_value.shape[1],rbg_value.shape[2]))
        for n, colors_array in enumerate(colors_arrays):
            dist = np.sum( (np.transpose(rbg_value,(1,2,0)) - colors_array)**2, axis = 2 )
            dbz_index = np.where( (dist == dist.min()) & (dist.min()< 10) )
            #print(dist.min())
            dBZ_figure[dbz_index] = dBZ_levels[n] # -0.1 

        dBZ_figure = np.where(dBZ_figure < 10.0, np.nan , dBZ_figure)

        ## Convert to the aircraft-centered grids
        dist_y = (lat - self.lat)*111.0 # unit: km
        dist_x = np.deg2rad((lon - self.lon)) * R_earth * np.cos(np.deg2rad(lat)) # for np.arrays, * is for element-wise product

        R_grid = np.sqrt(dist_y * dist_y + dist_x * dist_x)
        # get the azimuth from 0 to 360 from east
        Az_grid = np.abs(np.arctan(dist_y / dist_x) * 180 / math.pi)
        Az_grid = np.where( (dist_x < 0) & ( dist_y > 0), 180 - Az_grid, Az_grid)
        Az_grid = np.where( (dist_x < 0) & ( dist_y < 0), 180 + Az_grid, Az_grid)
        Az_grid = np.where( (dist_x > 0) & ( dist_y < 0), 360 - Az_grid, Az_grid)
        print(Az_grid.max(),Az_grid.min())

        ngate = int(np.round( (self.Rmax_Air-self.Rmin_Air) * 1000 / self.Bin_length ))
        Azmin = self.heading_angle  - self.range_angle 
        Azmax = self.heading_angle  + self.range_angle 

        # Calculate the coverage of the radial beam of the current carrier in polar coordinate
        dbz_ac = np.zeros((360, ngate))
        hid_ac = np.zeros((360, ngate))
        longtitude = np.zeros((360, ngate))
        latitude = np.zeros((360, ngate))
        mask_Z = np.zeros((dBZ_figure.shape[0],dBZ_figure.shape[1]))
        radarX = np.zeros((360, ngate))
        radarY = np.zeros((360, ngate))

        for i_az in np.arange(Azmin, Azmax+1): #azimuth comes from east, the direction is counterclockwise
            for i_bin in np.arange(0, ngate):
                i_az_valied = int(self.AngleToValid(i_az))

                x = i_bin * self.Bin_length * np.sin(np.deg2rad(i_az))
                y = i_bin * self.Bin_length * np.cos(np.deg2rad(i_az))
                #radarX.append(x)
                #radarY.append(y)
                radarX[i_az_valied, i_bin]  = x
                radarY[i_az_valied, i_bin]  = y

                lat_grid = y / 1000.0 /111.0 + self.lat
                lon_grid = np.rad2deg(x / 1000.0 /(R_earth * np.cos(np.deg2rad(lat_grid)))) + self.lon
                y_index = np.argmin(np.abs(lat[:, 10] - lat_grid)) #10 is a random number, 因为是等经纬度投影
                x_index = np.argmin(np.abs(lon[10, :] - lon_grid))
                
                dbz_ac[i_az_valied, i_bin] = dBZ_figure[y_index, x_index] 
                longtitude[i_az_valied, i_bin]  = lon_grid
                latitude[i_az_valied, i_bin]    = lat_grid
                
                mask_Z[y_index, x_index] = 1

        longtitude = np.where(longtitude == 0.0, np.nan, longtitude)
        latitude = np.where(latitude == 0.0, np.nan, latitude)
        return radarX, radarY, dbz_ac

    def save_dbz(self, in_file):
        raw_img_array = plt.imread(in_file)
        #0-255整数形式的RGB数值更为精确和通用，plt.imread读取的颜色为0-1的浮点数，因此转换为int型
        rgb_img_array = (raw_img_array * 255).astype(int)
        #将底图与dbz颜色分离
        #初始化阶段将原始图片数组分别存入两个数组。
        #存dbz的数组为data_img_array
        flaw_img_array = copy.deepcopy(rgb_img_array)
        #存底图的数组为flaw_img_array，由于后期底图的数组主要用于对缝隙的填补，因此使用flaw。
        data_img_array = copy.deepcopy(rgb_img_array)

        # 方法2: 先将图中的dbz剔除掉，得到纯净的底图，然后获取底图带颜色区域的坐标，利用该坐标将底图赋值为白色，从而获取dbz
        # 1. 剔除dbz像素点，提取出底图部分的坐标
        for colors_array in colors_arrays:
            dist = np.sum((rgb_img_array - colors_array) ** 2, axis=2)
            dbz_index = np.where(dist==dist.min())
            flaw_img_array[dbz_index] = np.array([255,255,255])

        flaw_index = np.where(flaw_img_array.sum(axis=2)<255*3) #非白色的坐标位置，即底图的坐标位置

        # 2. 将底图部分赋值为空白
        data_img_array[flaw_index] = np.array([255,255,255])

        data_img_array[upper_y:lower_y+1,left_x:right_x+1] = np.array([255,255,255])

        #以上图片中都有dBZ裂痕，下面进行填补
        # 1. 获取裂痕坐标
        flaw_yx = np.where(flaw_img_array.sum(axis=2)<255*3) #和方法2中的思路一样
        # 2. create a mask
        mask = np.full(raw_img_array.shape[:2], False)
        mask[flaw_yx] = True
        # 3. use scipy.spatial.KDTree to look for the nearest value to fill the cracks
        flaw_xy = flaw_yx[::-1]
        data_xy = np.where(~mask)[::-1]

        data_points = np.array(data_xy).T
        flaw_points = np.array(flaw_xy).T

        data_img_array[mask] = data_img_array[~mask][KDTree(data_points).query(flaw_points)[1]] #返回值是离查询点最近的点的距离和索引
        #save the image array to png
        plt.imsave("dbz.png",np.uint8(data_img_array))
    
    def AngleToValid(self, angle):
            ans = angle - int(np.floor_divide(angle, 360.0)) * 360
            return ans

if __name__=='__main__':

    infile = 'Z_RADA_C_BABJ_20220716015400_P_DOR_RDCP_R_ACHN.PNG'
    ac_parameters = { "Rmax_Air":240, "Rmin_Air":0, "heading_angle":-8.0365, "ll_center":[29.6682,111.5598], "range_angle":90, "Bin_length":150 }
    gac = Ground2AC(infile, ac_parameters)
    gac.plot_range()
    gac.plot_ground2ac_white()
    gac.plot_ground2ac_black()
