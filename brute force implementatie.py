import numpy as np # imports
import matplotlib.pyplot as plt

data = np.loadtxt('data2.csv', delimiter=',')





def get_distance(x1, x2): 

    sum = 0
    for i in range(len(x1)):
        sum += (x1[i] - x2[i]) ** 2
    return np.sqrt(sum)

def get_similarity(x1, x2, parameter):
    similarity = np.exp((-get_distance(x1, x2)**2)/parameter)
    return similarity

def get_L(f_s,pm): 
    W = []
    for i in range(len(data)):
        r = []
        for j in range(len(data)):
            r.append(f_s(data[i], data[j], pm))
        W.append(r)
    rowsums = []
    for i in range(len(data)):
        rowsums.append(sum(W[i]))
    D = np.diagflat(rowsums)
    L = D - W
    return L



def get_U(w,v,n):
    idx = w.argsort()
    indices = idx[:n]
    U = []
    for i in range(len(data)):
        row = []
        for j in indices:
            row.append(v[i,j])
        row.append(i)
        U.append(row)
    return U



def get_distance_u(u1, u2):
    sum = 0
    for i in range(k):
        sum += (u1[i] - u2[i]) ** 2
    return np.sqrt(sum)

def circle_class(data):
    
    per = 0.6
    
    r_list = []
    
    for d in data:
    
        r = np.sqrt(d[0]**2+d[1]**2)
        
        if r< 0.5:
            r_list.append("a")
            
        if r >= 0.5 and r < 0.85:
            
            r_list.append("b")
            
        else: 
            r_list.append('c')
            
    a,b,c = 0,0,0
    
    for k in r_list:
        
        if k == "a":
            
            a+=1
            
        if k == "b":
            
            b+=1
        
        if k == "c":
            
            c+=1
            
    norma,normb,normc = a/(a+b+c),b/(a+b+c),c/(a+b+c)
    
    print(norma,normb,normc)
            
    if norma > per or normb> per or normc > per:
        
        return True
    
    else:
        return False
        
    
max_iters = 1000

classcheck = True
while classcheck: 
    sig = 10**(np.random.uniform(-3,-1))
    
    #sig = np.round(sig,int(-np.log10(sig)+3))
    
    print(sig)

    L = get_L(get_similarity,sig)

    w,v = np.linalg.eig(L) 

    w,v = np.linalg.eig(L)
    idx = w.argsort()
    
    k = 3 

    U = get_U(w,v,k)
    
    list = np.random.choice(range(len(U)), k, replace=False) 


    centroids = []
    for i in list:
        centroids.append(U[i])


    temp = centroids[:]

    
    converged = False
    current_iter = 0 

    try:
        while (not converged) and (current_iter < max_iters):
        
            cluster_list = [[] for i in range(len(centroids))] 
        
            for x in U:  
                distances_list = []
                for c in centroids:
    
                    distances_list.append(get_distance_u(c, x)) 
                cluster_list[int(np.argmin(distances_list))].append(x)
        
            prev_centroids = centroids[:]
        
            centroids = []
        
            for j in range(len(cluster_list)):
                centroids.append(np.mean(cluster_list[j], axis=0))
        
            pattern = np.abs(np.sum(prev_centroids) - np.sum(centroids)) 
        
            print('K-MEANS: ', int(pattern))
        
            converged = (pattern == 0)
        
            current_iter += 1
            
            
        checks = []
            
        for i in range(len(cluster_list)):
            c = np.array(cluster_list[i])[:,k]
            d = c.astype(int)
            e = data[d]
                
            sc = circle_class(e)
                
            print('cluster '+str(i)+' check = ' +str(sc))
                
            checks.append(sc)
                
        if False not in checks:
                
            classcheck = False
            
    except:
        None
            
        
        
        
        


colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
temp = np.array(temp)
cluster_list = np.array(cluster_list)
plt.figure()
for i in range(len(cluster_list)):
    c = np.array(cluster_list[i])[:,k]
    d = c.astype(int)
    e = data[d]
    plt.scatter(e[:,0],e[:,1],color = colors[i])

    
    plt.xlabel('X')
    plt.ylabel('Y')

    
def circle(r):
    
    x = np.arange(-r,r,0.001)
    
    ypl = np.sqrt(r**2-x**2)
    
    ymi = -np.sqrt(r**2-x**2)
    
    return np.concatenate((x,x)),np.concatenate((ypl,ymi))

plt.figure()
for i in range(len(cluster_list)):
    c = np.array(cluster_list[i])[:,k]
    d = c.astype(int)
    e = data[d]
    plt.scatter(e[:,0],e[:,1],color = colors[i])

    
    plt.xlabel('X')
    plt.ylabel('Y')

c1 = circle(0.5)
c2 = circle(0.85)
plt.scatter(c1[0],c1[1],c='k',s=0.5)
plt.scatter(c2[0],c2[1],c='k',s=0.5)
    
plt.savefig("circles-clustered.png")
plt.plot()
plt.show()



