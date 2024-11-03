#This the code for my paper: https://arxiv.org/abs/2410.19987
#(c) M. Andrecut (2024)
import numpy as np
from skimage.transform import resize
from tensorflow import keras

def normalize0(x):
    (N,L,L) = np.shape(x)
    x = np.sqrt(x) # optional
    for n in range(N):
        x[n] -= np.mean(x[n])
        x[n] = x[n]/np.linalg.norm(x[n])
    x = np.reshape(x,(N,L*L))
    return x

def normalize(x):
    (N,L,L) = np.shape(x)
    x = np.sqrt(x)  # optional   
    f = np.zeros((N,L,L//2+1),dtype="float32") 
    for n in range(N):
        x[n] -= np.mean(x[n])
        f[n] = np.abs(np.fft.rfft2(x[n])-np.mean(x[n]))
        x[n] = x[n]/np.linalg.norm(x[n])
        f[n] -= np.mean(f[n])        
        f[n] = f[n]/np.linalg.norm(f[n])    
    x = np.reshape(x,(N,L*L))
    f = np.reshape(f,(N,L*(L//2+1)))        
    return np.hstack((x,f))/np.float32(np.sqrt(2))          

def vclass(xl,K):
    r = np.zeros((len(xl),K)).astype("float32")
    for n in range(len(r)):
        r[n,xl[n]] = 1
    return r
    
def skiresize(x,s):
    (N,l,l) = np.shape(x)
    y = np.zeros((N,int(s*l),int(s*l)),dtype="float32")
    for n in range(len(x)):
        y[n] = resize(x[n],(int(s*l),int(s*l)))
    return y
    
def random_weights(M,q):
    v = np.random.randn(M,int(M*q)).astype("float32")
    for i in range(M*q):
        v[:,i] = v[:,i]/np.linalg.norm(v[:,i])
    v = np.sqrt(M)*v
    q = np.ones((M,1)).astype("float32")
    return np.hstack((v,q))    

def orthogonal_random_weights(M):
    v = np.random.randn(M,int(M)).astype("float32")
    v = np.linalg.qr(v)[0].T
    return v

if __name__ == "__main__":     
    np.random.seed(123457)    
    T,s,q = 15,2,2 # T=iterations, s=image resize, q=projections scaling
    a = 0 # regularizetion a=0 for MNIST, a=10 for fMNIST

    (x, xl), (y, yl) = keras.datasets.mnist.load_data() # MNIST
#    (x, xl), (y, yl) = keras.datasets.fashion_mnist.load_data() #fMNIST

    x,y = x.astype("float32"),y.astype("float32") # convert to float
    x,y = skiresize(x,s), skiresize(y,s) # resize images
    x,y = normalize(x),normalize(y) # normalize images
    (N,M),(NN,M),K = np.shape(x),np.shape(y),np.max(xl)+1        
    print("N=",N,"M=",M*q,"s=",s,"q=",q,"a=",a) # print parameters
    v = orthogonal_random_weights(M) # perform orthogonal projections
    x,y = np.dot(x,v),np.dot(y,v) #        
    r = vclass(xl,K) # one-hot encoding classes
    h = np.zeros((NN,K),dtype="float32") # cummulative results
    for t in range(T): # RRNN 
        v = random_weights(M,q)
        f = np.tanh(np.dot(x,v))
        w = np.linalg.solve(np.dot(f.T,f) + a*np.identity(int(M*q+1)).astype("float32"),np.dot(f.T,r))        
        r -= np.dot(f,w)/2
        f = np.tanh(np.dot(y,v))
        h += np.dot(f,w)
        p = np.round(100*np.sum(np.argmax(h,axis=1)==yl)/NN,3)
        rr = np.linalg.norm(r)
        print(t,"p=",p,"%","|r|=",rr,"a=",a)

