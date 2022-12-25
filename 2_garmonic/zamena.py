import numpy as np
from requests import patch
from scipy import integrate
import matplotlib.pyplot as plt
from scipy.optimize import root
import joblib 
from numpy import linalg as LA
import os, shutil


#Параметры системы 

N_JOB = 4
col_razb = 10
MAX_GRAPH = 50
eps = 0.1

class Equilibrium_states(object):
    
    #инициализация системы
    def __init__(self,p = [3,1]):
        self.N, self.m  = p
        self.M = 0
        self.k1 = 1
        self.k2 = 1
        self.alpha = 0
        self.beta = 0
        self.sost = []
        self.ust = []
        self.un_ust = []
        self.t = np.linspace(0,100,100)

    # Сама система (возможно стоит использовать при расчете состояний равновесия)
    def syst(self,param,t):
        N = self.N
        M = self.M
        alpha = self.alpha
        beta = self.beta
        m = self.m
        k1 = self.k1
        k2 = self.k2
        
        x,w = param #[x,y,w,v] - точки
        f = np.zeros(2)
        # x with 1 dot
        f[0] = 1/m*(1/N * ( N*(np.sin(alpha) + np.sin(beta)) + (N - M)*(k1*np.sin(-x-alpha) + k2*np.sin(-2*x-beta)) - M*(k1*np.sin(x-alpha) + k2*np.sin(2*x - beta))))
        # x with 2 dots
        f[1] = w
        
        return f
    
    #поиск состояний равновесия для определенного параметра
    def state_equil(self,NMAB=[3,1,np.pi/2,np.pi/2]):
        all_sol = [] 
        self.N,self.M,self.alpha,self.beta = NMAB
        ran_x=[0,2*np.pi]
        
        X = np.linspace(ran_x[0],ran_x[1],col_razb)
        w = 0
        
        for x in X:
            sol = root(self.syst,[x,w],args=(0),method='lm')
            xar,war = sol.x
            xar = round(xar,6)
            if [xar,self.M,self.alpha,self.beta] not in all_sol and (xar>=0 and xar<2*np.pi):   
                all_sol.append([xar,self.M,self.alpha,self.beta])
                
        return all_sol
    
    #поиск всех состояний равновесия
    def parall_st_eq(self):
        N = self.N
        M = np.linspace(1,N-2,N-2, dtype = 'int')
        
        alpha = np.linspace(0,np.pi,4)
        beta = np.linspace(0,np.pi,4)
        self.sost = joblib.Parallel(n_jobs = N_JOB)(joblib.delayed(self.state_equil)([N,m,al,bt]) for m in M for al in alpha for bt in beta if m<N)
        
        self.rec_st_and_unst_st_eq()
        
        self.save_all_sost()
        
    
    def save_all_sost(self):
        name = f"2_garmonic\\res\\n_{self.N}\\res_n_{self.N}.txt"
        
        with open(name,"w",encoding="utf-8") as file: 
            for tmp in self.sost:
                for j in tmp:
                    j = str(j)+'\n'
                    file.write(j)  
                # file.write('--------------------------------------------------------\n')
    #матрица якоби
    def jakobi(self, param):
        x,M,alpha,beta = param
        N = self.N
        m = self.m
        k1 = self.k1
        k2 = self.k2
        f = []
        f.append([0, 1])
        f.append([1/(m*N) *((N-M)*(-k1*np.cos(x+alpha) - 2*k2*np.cos(2*x+beta)) - M*(k1*np.cos(x-alpha) + 2*k2*np.cos(2*x-beta))), -1/m])
        arr = np.array(f)
        return(arr)

    #поиск собственных чисел при определененных параметрах
    def eigenvalues(self,param):
        matrix = self.jakobi(param)
        lam, vect = LA.eig(matrix)
        return lam
    
    #Сохранение устойчивых и неустойчивых состояний равновесия 
    def rec_st_and_unst_st_eq(self):
        name_s = f"2_garmonic\\res\\n_{self.N}\\stable_{self.N}.txt"
        name_n = f"2_garmonic\\res\\n_{self.N}\\non_stable_{self.N}.txt"
        name_o = f"2_garmonic\\res\\n_{self.N}\\range_zero_{self.N}.txt" 
        self.create_path(name_s[:18])
        self.create_path(name_n[:18])          
        with open(name_s,"w",encoding="utf-8") as file_s: 
            with open(name_n,"w",encoding="utf-8") as file_n:
                with open(name_o,"w",encoding="utf-8") as file_o: 
                    for i in self.sost:
                        for j in i:
                            tmp = self.eigenvalues(j)
                            z = 0
                            o = 0
                            for l in tmp.real:
                                if  l>-0.0005 and l<0.0005:
                                    o = 1
                                    break 
                                
                                if l < 0:
                                    z+=1
                            text = str(j)+'\t'+str(tmp.real)+'\n' 
                            if z == 2 and o == 0:
                                file_s.write(text)
                            elif o==1:
                                file_o.write(text)
                            else:
                                file_n.write(text)
    
    #динамика для одной точки
    def dinamic(self,params = [2.094395, 1,  2.0943951023931953, 2.0943951023931953]):
        start_point=np.zeros(2)
        start_point[0], self.M, self.alpha, self.beta = params 
        start_point[0] = start_point[0]+eps
        
        tmp = integrate.odeint(self.syst, start_point, self.t)
        plt.plot(self.t,tmp[:,0],label="x")
        # plt.xlim(0, 100)
        # plt.ylim(-10, 10)
        plt.legend()
        plt.show()
    
    #динамика, но сохраняем
    def rec_dinamic(self,way,z=1,params = [2.094395, 1, 2.0943951023931953,2.0943951023931953]):
        start_point=np.zeros(2)
        start_point[0],self.M,self.alpha,self.beta = params 
        start_point[0] += eps
        tmp = integrate.odeint(self.syst, start_point, self.t)
        plt.plot(self.t,tmp[:,0],label="x")
        plt.xlim(0, 100)
        plt.ylim(-10, 10)
        plt.legend()
        plt.savefig(way + f'graph_{z}.png')
        plt.clf()

    #показываем графики, но выбираем какие и отдельно в папочки
    #ключевые слов "all", "st", "un_st"
    def show_sost(self,key = 'all'):
        n = self.N
        name = "2_garmonic\\res\\n_\\"
        way = name
        
        if key == 'st':
            name = name+"stable_.txt"
            way = way+"stable\\"
        elif key == "un_st":
            name = name + "non_stable_.txt"
            way = way+"unstable\\"
        elif key == "rz":
            name = name + "range_zero_.txt"
            way = way+"range_zero\\"
        elif key == "all":
            name = name + "res_n_.txt"
            way = way+"all\\"
        else:
            print("wrong key")
            return
            
        sdvig1 = -4
        sdvig2 = 17
        
        way_or = 'zamena\\'
        
        name = name[0:sdvig1]+f"{n}"+name[sdvig1:]
        name = name[0:sdvig2]+f"{n}"+name[sdvig2:]
        # way = way[0:sdvig1]+f"{n}"+way[sdvig1:]
        way = way[0:sdvig2]+f"{n}"+way[sdvig2:] + way_or

        self.create_path(way)
        self.clean_path(way)
        # print(way)
        
        arr  = self.razbor_txt(name)
        
        rang = len(arr)
        if rang > MAX_GRAPH:
            rang = MAX_GRAPH
            
        for i in range(rang):
            self.rec_dinamic(params = arr[i],way = way,z=i+1)

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
            

    def razb_str(self, str):
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

if __name__ == "__main__":
    tmp = [4,1]
    es = Equilibrium_states(p = tmp)
    es.dinamic(params=[6.103964, 1, 2.0943951023931953, 3.141592653589793])
    
    # es.parall_st_eq() #подсчет всех состояний
    
    # # es.show_sost(key='all') #сохранение графиков #ключевые слов "all", "st", "un_st","rz"
    
    # tmp = ['st','un_st','rz']
    # for i in tmp:
    #     es.show_sost(key=i) #сохранение графиков #ключевые слов "all", "st", "un_st","rz"
    
    
  