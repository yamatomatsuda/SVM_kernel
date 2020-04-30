import numpy as np
import matplotlib.pyplot as plt

"""
ソフトマージンSVM
"""

class MySVM:
    
    
    def __init__(self,C=1.0,episode=100,linear_constant=1.0,degree=2):
        self.alpha = 0.1
        self.support_vector_number=[]
        self.episode = episode
        self.omega = None
        self.sita = None
        self.C=C
        self.linear_constant=linear_constant
        self.degree = degree

    
    def get_first_lamda(self):
        """lambdaの初期値設定"""
        self.lamda = np.random.uniform(1,0,len(self.X))
        
    
    
    def get_omega(self):
        """omegaを出す"""
        self.omega = self.lamda*self.Y
        #print(self.omega)
    def get_sita(self):
        """サポートベクターの中でも、教師ラベルが正と負のものがある
        それぞれの値の偏りを無くすため、正の教師ラベルの領域、負の教師ラベルの領域から
        もし存在すればそれらの平均によってバイアス値self.sitaを定義する
        """
        
        _p=[]
        _n=[]
        for _ in self.support_vector_number:
            if self.Y[_]==1:
                _p.append(_)
            else:
                _n.append(_)
        
        """
        if (len(_p) and len(_n)) !=0:
            self.sita = -0.5 * (np.dot(self.omega.T,self.X[_p[-1]])+np.dot(self.omega.T,self.X[_n[0]]))
        else:
            self.sita = self.Y[self.support_vector_number[-1]]-np.dot(self.omega.T,self.X[self.support_vector_number[-1]])
        """
        _kernel_temp = 0
        for _ in self.support_vector_number:
            _kernel_temp = _kernel_temp +self.omega[_]*self.kernel(self.X[_],self.X[self.support_vector_number[-1]])
        self.sita=self.Y[self.support_vector_number[-1]]-_kernel_temp
    
    def read_csv(self):
        """読み込むデータによって変える"""
        
        self.X = np.loadtxt(fname="data1.csv",dtype='float',skiprows=5,usecols=(4,7),delimiter=',')
        self.Y = np.loadtxt(fname="data1.csv",dtype= 'unicode',skiprows=5,usecols=(1),delimiter=',')
        
    
        for _ in range(len(self.Y)):
            if '雨' in self.Y[_]:
                self.Y[_] = 1
            
            else:
                self.Y[_] = -1
        
        #self.X = np.loadtxt(fname="testdata.csv",dtype='float',skiprows=1,usecols=(0,1),delimiter=',')
        #self.Y = np.loadtxt(fname="testdata.csv",dtype= 'float',skiprows=1,usecols=(2),delimiter=',')
        self.Y = self.Y.astype(float)
    
    
    def standardzation(self):
        """元のデータの正規化"""
        self.raw_X = self.X
        self.X = (self.X-self.X.mean(keepdims=True))/self.X.std(keepdims=True)
    
    
    def restandardzation(self):
        """正規化データを元のデータに戻す"""
        self.X = self.raw_X
    
    
    def limit_lamda(self):
        """制約条件の「ラグランジュ乗数と教師ラベルの掛け算の和が0」を満たすための正規化

        教師ラベルの正負でラグランジュ乗数を分けてそれぞれで和が１になるようにそれぞれ
        一つ一つのデータを
        正の教師ラベルを持つlambdaの値の和、負の教師ラベルを持つlambdaの値の和で割る
        """
        _sum1 = 0
        _sum2 = 0
        for _ in range(len(self.Y)):
            if self.Y[_]==1:
                _sum1 = _sum1 + self.lamda[_]*self.Y[_]
            else:
                _sum2  = _sum2 + self.lamda[_]*(-self.Y[_])
        
  

        for _ in range(len(self.Y)):
            if self.Y[_]==1:
                self.lamda[_] = self.lamda[_]/_sum1
            else:
                self.lamda[_] = self.lamda[_]/_sum2
        
    
    def gradient_descent(self):
        """更新式"""
        for _i in range(len(self.lamda)):
            _sum = 0
            for _j in range(len(self.Y)):
                _sum = _sum + self.lamda[_j]*self.Y[_i]*self.Y[_j]*self.kernel(self.X[_j].T,self.X[_i])
            self.lamda[_i] = self.lamda[_i] + self.alpha*(1-_sum)

    def kernel(self,x1,x2):
        return (self.linear_constant+np.dot(x1,x2))**self.degree

    def main(self):
        """main文"""
        """まずはデータの整形"""
        self.read_csv()
        self.standardzation()
        self.get_first_lamda()
        
        """
        最急降下法によって正規化
        毎回更新時にlambdaを制約条件に当てはまるように正規化
        """
        for _  in range(self.episode):
            
                self.limit_lamda()
                
                self.gradient_descent()

        self.limit_lamda()
        self.restandardzation()

        """サポートベクターの抽出"""
        for _ in range(len(self.lamda)):
            if self.lamda[_]>0 and self.lamda[_]<=self.C:
                self.support_vector_number.append(_)

        """omegaとsitaを出す"""
        self.get_omega()
        self.get_sita()

        
if __name__ == "__main__":
    
    """Cのデフォルト値は1.0、最急降下法の繰り返しのデフォルト値は100"""
    svm=MySVM(C=1,episode=10,degree=3)
    svm.main()

    #print(svm.lamda)
    #print(svm.omega,svm.sita)
    
    #for _ in svm.support_vector_number:
        #print(svm.X[_])
   
 
    x_min = 30
    x_max = 100
    y_min = 0
    y_max = 40
    Z=[]

    x = np.linspace(x_min,x_max,200)
    y = np.linspace(y_min,y_max,200)
    XX,YY = np.meshgrid(x,y,sparse=True)

    for array_x in XX.T:
        for array_y in YY:
            #print(np.concatenate([array_x,array_y]))
            for _ in svm.support_vector_number:
                _sum = 0
                _sum += svm.omega[_]+svm.kernel(np.concatenate([array_x,array_y]).T,svm.X[_])
            Z.append(_sum+svm.sita)
  
    plt.scatter(svm.X[:,0], svm.X[:,1],s=80, facecolors='none')
    plt.scatter(svm.X[:, 0], svm.X[:, 1], c=svm.Y)
    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = np.asarray(Z)
    #Zの値が大きすぎる
    Z = Z.reshape(XX.shape)
    print(Z)
    plt.contour(XX, YY, Z,levels=[5, 0, 5], colors='k', linewidths=1, origin='lower')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

 
    plt.axis('tight')

    plt.show()

