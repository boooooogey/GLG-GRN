import numpy as np
import h5py
from IPython import embed
from scipy.io import loadmat
#from FISTA_SOLVER import training
import glmnet_python
from glmnet import glmnet
from glmnetSet import glmnetSet

def mat_matrix(data,ir,jc):
    X = np.zeros((np.max(ir)+1,len(jc)-1))
    for n,i in enumerate(jc[1:]):
        X[ir[jc[n]:i],n] = data[jc[n]:i]
    return X

def read_matlab(name):
    '''
    Read MATLAB's .mat files
        name: Path to the file
    '''
    file = h5py.File(name,'r')
    ir = np.array(file.get('X/ir'),dtype=int)
    jc = np.array(file.get('X/jc'),dtype=int)
    data = np.array(file.get('X/data'))
    ptime=np.array(file.get('ptime')).reshape(-1)
    X = mat_matrix(data,ir,jc)
    file.close()
    return X, ptime 

def gram_matrix(ptime,L,Dt,kernel,mask):
    '''
    Gives the gram for the given kernel function
        ptime: irregular times
        L: Time lag to be studied
        Dt: average sampling interval length
    '''
    NL = int(L/Dt)
    #ii = np.where(ptime > L)[0]
    K = np.empty((NL,len(ptime),len(ptime)))
    Ksum = np.empty((NL,mask.shape[1],mask.shape[0]))
    for n,i in enumerate(reversed(range(1,NL+1))):
        tmp = ptime - i * Dt
        K[n] = kernel(tmp.reshape(-1,1),ptime.reshape(1,-1))
        Ksum[n] = K[n] @ np.logical_not(mask).T
    return K,Ksum

def convert_to_FISTA(X,ptime,kernel,L,Dt,g,mask,queue):
    '''
    Converts GLG problem into a FISTA problem
        X: data matrix
        ptime: irregular time points
        kernel: kernel function to be used for the GLG
        L: Time Lag to be studied
        Dt: average sampling interval length
    '''
    Ngenes, Ntimes = X.shape
    K,Ksum = gram_matrix(ptime,L,Dt,kernel,mask)
    NL = int(L/Dt)
    iiy = ptime > L
    Ny = np.sum(iiy)
    remind = np.where(np.logical_and(np.logical_not(mask[0,:]),ptime > L))[0]
    A = np.empty((len(remind),NL * Ngenes)) 
    tmp = X[queue,:] * np.logical_not(mask)
    for i in range(NL):
        norm = Ksum[i,remind,:]
        A[:,np.arange(Ngenes)*NL+i] = K[i, remind, :] @ tmp.T / norm[:,queue]
    tmp = X[queue,:]
    y = tmp[g,remind]
    return A,y,K

def print_smat(m):
    for c,r in zip(*np.where(m.T != 0)):
        print("({},{})\t\t{}".format(r+1,c+1,m[r,c]))

def print_dif_smat(m1,m2,thr):
    for c,r in zip(*np.where(np.abs(m1-m2).T > thr)):
        print("({},{})\t\t{} - {}".format(r+1,c+1,m1[r,c],m2[r,c]))

def reformat_output(w,ngene,NL,g,adj_matrix,queue):
    p = np.eye(ngene)[np.argsort(queue),:]
    for i in range(w.shape[1]):
        res = p @ np.sum(w[:,i].reshape(ngene,NL),axis=1)
        adj_matrix[i,:,g] = res

def single_gene_GLG(X,ptime,L,Dt,g,sigma,lambdas,mask,queue):
    k = lambda x,y: np.exp(-np.power(x-y,2)/sigma)
    A,y,K = convert_to_FISTA(X,ptime,k,L,Dt,g,mask,queue)
    nSamp, nFea = A.shape
    NL = int(L/Dt)
    #out = training(A,y,np.random.rand(A.shape[1]),0)
    opt = glmnetSet()
    opt['lambdau'] = lambdas
    opt['nlambda'] = 5
    opt['maxit'] = 10000
    opt['penalty_factor'] = np.ones(nFea) 
    opt['penalty_factor'][g*NL:(g+1)*NL] = 0
    fit = glmnet(x=A,y=y.reshape(-1),family='gaussian',**opt)
    #if fit['beta'].shape[1] < len(lambdas):
    #    opt['nlambda'] = 1
    #    opt['lambdau'] = np.array([lambdas[0]])
    #    fit1 = glmnet(x=A,y=y.reshape(-1),family='gaussian',**opt)
    #    beta = fit1['beta']
    #    for i in lambdas[1:]:
    #        opt['lambdau'] = np.array([i])
    #        fit1 = glmnet(x=A,y=y.reshape(-1),family='gaussian',**opt)
    #        beta = np.column_stack((beta,fit1['beta']))
    #    fit['beta'] = beta
    return fit

def GLG(X,ptime,L,Dt,lambdas,sigma,prob):
    nGene, ntime = X.shape
    adj_matrix = np.zeros((len(lambdas),nGene,nGene))
    queue = np.arange(nGene)
    mask = drop_zero(X,prob)
    for g in range(nGene):
        fit = single_gene_GLG(X,ptime,L,Dt,0,sigma,lambdas,mask,queue)
        reformat_output(fit['beta'],nGene,int(L/Dt),g,adj_matrix,queue)
        if g+2 <= nGene:
            queue = np.arange(nGene)
            queue[1:g+2] = queue[:g+1]
            queue[0] = g+1
    return adj_matrix

def drop_zero(X,prob):
    mask = np.random.rand(*X.shape)
    return np.logical_and(mask < prob,X == 0)

def borda_voting(adj):
    L = adj[0].size
    weigths = np.power(1/(np.arange(L)+1),2)
    sum = np.zeros(L)
    for i in range(adj.shape[0]):
        term = np.zeros(L)
        ranking = adj[i].reshape(-1)
        sorted_ii = np.argsort(np.abs(ranking))[::-1]
        term[sorted_ii] = weigths 
        sum += term * (ranking != 0)
    agg = sum.reshape(adj.shape[1],adj.shape[2])
    np.fill_diagonal(agg,0)
    return agg

def create_edge_list(agg,names):
    n = np.array(names)
    L = agg.shape[0]
    absagg = np.abs(agg)
    size = agg.size
    weights = agg.reshape(-1)
    #rank = np.argsort(weights)[::-1]
    #rank = size - np.argsort(ii) - 1
    t,r = np.mod(np.arange(size),L),np.floor_divide(np.arange(size),L) 
    return list(reversed(sorted(zip(n[r],n[t],weights),key= lambda x: x[2])))

def create_influence_list(agg,names):
    inf = np.sum(agg,axis=1)
    return list(reversed(sorted(zip(names,inf),key= lambda x: x[1])))

def format_table(table,width=15):
    l = len(table)
    w = len(table[0])
    t = []
    t.append((("{:^"+str(width)+"}|")*(w-1) + "{:^"+str(width)+"}").format('Regulator','Target','Score'))
    temp = ("{:^"+str(width)+"}|")*(w-1) + "{:^"+str(width)+".5f}"
    t.append("_"*(width*w+w-1)) 
    for i in table:
        t.append(temp.format(*i))
    return t

def table_write(cont,path):
    with open(path,'w') as file:
        for i in cont:
            file.write(i+'\n')

def main():
    L = 15  # Lag
    Dt = 3  # To discretize the time
    sigma = 0.5  # sigma for the gaussian kernel
    g = 0  # The gene that's being predicted
    lambdas = np.array([0.1,0.05,0.02,0.01,0])
    family_param = 'gaussian'
    prob = 0.1
    #X, ptime = read_matlab('/data/src/SINGE/data1/X_SCODE_data.mat')
    #gene_names = [i[0] for i in (loadmat("/data/src/SINGE/data1/gene_list.mat")['gene_list']).reshape(-1)]
    data = np.load("/data/causal/data/simulated_dataset.npz",allow_pickle=True)
    X,ptime,gene_names = np.asarray(data['mat'].ravel()[0].todense()),data['ptime'].reshape(-1),data['gene_names'].reshape(-1)
    #X = X[[not i.startswith("HK") for i in gene_names],:]
    adj = GLG(X,ptime,L,Dt,lambdas,sigma,prob)
    agg = borda_voting(adj)
    table_edge = create_edge_list(agg,gene_names)
    table_influence = create_influence_list(agg,gene_names)
    tedge = format_table(table_edge)
    tinf = format_table(table_influence)
    table_write(tedge,"/data/causal/data1_output_edge.txt")
    table_write(tinf,"/data/causal/data1_output_inf.txt")

main()
