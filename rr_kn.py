import numpy as np
from skimage.transform import resize
from tensorflow import keras

def normalize0(x): # simple normalization
    (N,L,L) = np.shape(x)
    for n in range(N):
        x[n] -= np.mean(x[n])
        x[n] = x[n]/np.linalg.norm(x[n])
    x = np.reshape(x,(N,L*L))
    return x
    
def normalize(x): # image+fft normalization
    (N,L,L) = np.shape(x)  
    f = np.zeros((N,L,L//2+1),dtype="float32") 
    for n in range(N):
        x[n] -= np.mean(x[n])
        f[n] = np.abs(np.fft.rfft2(x[n]))
        x[n] = x[n]/np.linalg.norm(x[n])
        f[n] -= np.mean(f[n])        
        f[n] = f[n]/np.linalg.norm(f[n])    
    x = np.reshape(x,(N,L*L))
    f = np.reshape(f,(N,L*(L//2+1)))        
    x = np.hstack((x,f))/np.float32(np.sqrt(2))          
    return x    
    
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
    
def orthogonal_random_weights(M):
    v = np.random.randn(M,int(M)).astype("float32")
    return np.linalg.qr(v)[0].T

def kernel(y,u):
    N,J = len(y),len(u)
    q = np.dot(y,u.T)**5   
    return q
   
if __name__ == "__main__":     
    np.random.seed(123457)    
    T,s = 20,1 # T=iterations, s=image resize
    a = 0.01 # regularizetion a=0.01 for MNIST, a=0.05 for fMNIST
    J = 15000 # size of the kernel matrix  
  
    (x, xl), (y, yl) = keras.datasets.mnist.load_data() # MNIST
#    (x, xl), (y, yl) = keras.datasets.fashion_mnist.load_data() # fMNIST

    x,y = x.astype("float32"),y.astype("float32") # convert to float   
    x,y = skiresize(x,s), skiresize(y,s) # resize images        
    x,y = normalize(x),normalize(y) # normalize images        
    (N,M),(NN,M),K = np.shape(x),np.shape(y),np.max(xl)+1        
    print("J=",J,"N=",N,"M=",M,"s=",s,"a=",a) 
    v = orthogonal_random_weights(M) # perform orthogonal projections 
    x,y = np.dot(x,v),np.dot(y,v)        
    r = vclass(xl,K) # one-hot encoding classes
    ids = np.array([i for i in range(N)])
    h = np.zeros((NN,K),dtype="float32") # cummulative results
    for t in range(0,T): # RRKN
        np.random.shuffle(ids)
        u = x[ids[:J]]
        q = kernel(x,u)
        w = np.linalg.solve(np.dot(q.T,q) + a*np.identity(J,dtype="float32"),np.dot(q.T,r))
        r -= np.dot(q,w)
        q = kernel(y,u)        
        h += np.dot(q,w)       
        p = np.round(100*np.sum(np.argmax(h,axis=1)==yl)/NN,3)
        rr = np.linalg.norm(r)
        print(t,"p=",p,"%","|r|=",rr)

