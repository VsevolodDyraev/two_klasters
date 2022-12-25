import numpy as np
from requests import patch
from scipy import integrate
import matplotlib.pyplot as plt
from scipy.optimize import root
import joblib 
from numpy import linalg as LA
import os, shutil

#Параметры системы 

col_razb = 10
MAX_GRAPH = 50
eps = 0.1


class Original_sist(object):
    
    #инициализация системы
    def __init__(self,p = [3,1], fi=1):
        self.N, self.m,self.omega = p
        self.M = 0
        self.k1 = 1
        self.k2 = 1
        self.alpha = 0
        self.beta = 0
        self.fi1 = fi
        self.sost = []
        self.ust = []
        self.un_ust = []
        self.t = np.linspace(0,100,100)

    # Сама система (возможно стоит использовать при расчете состояний равновесия)
    def syst(self,param,t):
        N = self.N
        M = self.M
        omega = self.omega
        alpha = self.alpha
        beta = self.beta
        m = self.m
        k1 = self.k1
        k2 = self.k2
        
        
        fi1,fi2,v,w = param #[x,y,w,v] - точки
        f = np.zeros(4)
        # fi1 with 1 dot
        f[0] = 1/m*( 1/N * ( (-M) *(k1*np.sin(alpha) +k2*np.sin(beta)) + (N-M)*(k1*np.sin(fi2 - fi1 - alpha) + k2*np.sin(2*fi2 - 2*fi1 - beta)))  -v + omega)
        # fi2 with 1 dot
        f[1] = 1/m*( 1/N * ( -(N-M) *(k1*np.sin(alpha) +k2*np.sin(beta)) + M*(k1*np.sin(fi1 - fi2 - alpha) + k2*np.sin(2*fi1 - 2*fi2 - beta)))  - w + omega)
        # fi1 with 2 dots
        f[2] = v
        # fi2 with 2 dots
        f[3] = w
        
        return f
    
    #динамика для одной точки
    def dinamic(self, params = [np.pi, 1, np.pi/3, np.pi/3]):
        tmp = self.anti_zamena(arr=params)
        start_point=np.zeros(4)
        start_point[0],start_point[1], self.M,self.alpha, self.beta = tmp[0] 
        start_point[0] = start_point[0]+eps
        start_point[1] = start_point[1]+eps
        start_point[2] += eps
        start_point[3] += eps
        
        tmp = integrate.odeint(self.syst, start_point, self.t)
        plt.plot(self.t,tmp[:,0] - tmp[:,0],label="fi1")
        plt.plot(self.t,tmp[:,1] - tmp[:,0],label="fi2", linestyle = '--')
        # plt.xlim(0, 100)
        # plt.ylim(0, 100)
        plt.legend()
        plt.show()
    
    #динамика, но сохраняем
    def rec_dinamic(self,way,z=1,params = [1, 2.094395, 1, 2.0943951023931953,2.0943951023931953]):
        start_point=np.zeros(4)
        start_point[0],start_point[1], self.M,self.alpha, self.beta = params 
        start_point[0] += eps
        start_point[1] += eps
        tmp = integrate.odeint(self.syst, start_point, self.t)
        # for x in tmp:
        #     x[0] = np.sin()
        plt.plot(self.t,tmp[:,0] - tmp[:,0],label="fi1")
        plt.plot(self.t,tmp[:,1] - tmp[:,0],label="fi2", linestyle = '--')
        # plt.xlim(0, 100)
        # plt.ylim(0, 1)
        plt.legend()
        plt.savefig(way + f'graph_{z}.png')
        plt.clf()
        return tmp

    def rec_dinamic_par(self, way, z, arr):
        R1 = self.order_parameter(arr)
        plt.plot(self.t, R1)
        # plt.xlim(0, 100)
        plt.ylim(0, 1.1)
        plt.savefig(way + f'graph_{z}.png')
        plt.clf()

    #показываем графики, но выбираем какие и отдельно в папочки
    #ключевые слов "all", "st", "un_st"
    def show_sost(self,arr, key = 'all'):
        n = self.N
        name = "2_garmonic\\res\\n_\\"
        way = name
        
        if key == 'st':
            way = way+"stable\\"
        elif key == "un_st":
            way = way+"unstable\\"
        elif key == "rz":
            way = way+"range_zero\\"
        elif key == "all":
            way = way+"all\\"
        else:
            print("wrong key")
            return
            
        # sdvig1 = -4 
        sdvig2 = 17
        way_or = 'origin\\'
        way_par = 'order_params\\'

        # way = way[0:sdvig1]+f"{n}"+way[sdvig1:]
        way_p = way[0:sdvig2]+f"{n}"+way[sdvig2:] + way_par
        way = way[0:sdvig2]+f"{n}"+way[sdvig2:] + way_or
        self.create_path(way)
        self.clean_path(way)
        self.create_path(way_p)
        self.clean_path(way_p)
        # print(way)
                
        rang = len(arr)
        if rang > MAX_GRAPH:
            rang = MAX_GRAPH
            
        for i in range(rang):
            tmp = self.rec_dinamic(params = arr[i],way = way,z=i+1)
            self.rec_dinamic_par(way = way_p,z=i+1, arr = tmp)
             
    def sost_in_fi(self, key = 'all'):
        n = self.N
        name = "2_garmonic\\res\\n_\\"
        
        if key == 'st':
            name = name+"stable_.txt"
        elif key == "un_st":
            name = name + "non_stable_.txt"
        elif key == "all":
            name = name + "res_n_.txt"
        elif key == "rz":
            name = name + "range_zero_.txt"
        else:
            print("wrong key")
            return
        sdvig1 = -4 
        sdvig2 = 17
        
        name = name[0:sdvig1]+f"{n}"+name[sdvig1:]
        name = name[0:sdvig2]+f"{n}"+name[sdvig2:]

        res = self.razbor_txt(name)
        res_fi = self.anti_zamena(res)


        self.show_sost(arr = res_fi, key=key)
        # ress = self.order_parameter(res_fi)
        # self.show_sost(arr = ress, key=key)

    # просто рисовалка) 
    # def PlotOnPlane(self, arr):
    #     plt.plot()
    def anti_zamena(self, arr):
        ress = []
        fi1 = self.fi1
        for x in arr:
            fi2 = fi1 - x[0]
            ress.append([fi1, fi2, x[1], x[2], x[3]])
        return ress
                
    # чистим папку
    def clean_path(self,way):
        for filename in os.listdir(way):
            file_path = os.path.join(way, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
        
    #разбор txt-шников в float массив
    def razbor_txt(self,name):
        ress = []
        with open(name) as file:
            for line in file:
                ress.append(self.razb_str(line.rstrip()))
        return ress
        
    def change_N(self,N_):
        self.N = N_

    def create_path(self, way):
        if not os.path.exists(way):
            os.makedirs(way)
    def razb_str(self,str):
        all = []
        tmp = ''

        for c in str:
            if c==' ' or c=='[':
                continue
            if c==',' or c==']':
                all.append(float(tmp))
                tmp = ''
                if c==']':
                    break
                continue
            tmp+=c
        return all

    def order_parameter(self, arr):
        res = []
        for x in arr:
            sumr = 0
            sumi = 0
            for i in range(2):
                tmp = np.exp(x[i]*1j)
                sumr += tmp.real
                sumi += tmp.imag
            sum = 1/3 * np.sqrt(sumr ** 2 + sumi ** 2)
            res.append(sum)
        return res

if __name__ == "__main__":
    tmp = [4,1, 0]
    ors = Original_sist(p = tmp, fi = np.pi)
    # ors.dinamic(params=[[0.0, 2, 0.0, 3.141592653589793]])
    
    # ors.sost_in_fi(key='all') #"st","un_st","rz","all"
      
    tmp = ['st','un_st','rz']
    for i in tmp:
        ors.sost_in_fi(key=i) #"st","un_st","rz","all"
    
    

    # np.angel(fin - fi0)
    # параметр порядка

    # посмотреть 2х кластерное разбиение но со второй гармоникой