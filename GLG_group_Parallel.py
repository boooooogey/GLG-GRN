import multiprocessing as mp
import numpy as np
from IPython import embed
from ctypes import c_double
from GLG_group import *
import sys

def GLG_Parallel(inp):
    L, Dt, lamb, sigma, group_reg, probDrop, probZero, rep = inp
    X = np.frombuffer(X_buff).reshape(l,-1)
    ptime = np.frombuffer(ptime_buff)
    agg = np.frombuffer(agg_buff).reshape(l,-1)
    tmp = np.zeros_like(agg)
    for i in range(rep):
        tmp += GLG_network(X,ptime,L,Dt,lamb,sigma,group_reg,probDrop,probZero)
    lock.acquire()
    agg[:] = agg + tmp
    lock.release()

def init(lock_,X_buff_,ptime_buff_,agg_buff_,l_):
    global X_buff
    global ptime_buff
    global agg_buff
    global lock
    global l
    X_buff = X_buff_
    ptime_buff = ptime_buff_
    agg_buff = agg_buff_
    lock = lock_
    l = l_

def parse_hyperparameters(path):
    with open(path,'r') as file:
        hypes = file.readlines()
    hypes = [i.split("--") for i in hypes]
    list_hypes = []
    grs = np.linspace(0,1,num=len(hypes))
    for n,i in enumerate(hypes):
        lamb = 0
        group_reg = grs[n]
        Dt = float(i[2][3:])
        L = float(i[3][9:])
        sigma = float(i[4][13:])
        rep = int(i[6][10:])
        probZero = float(i[9][18:])
        probDrop = float(i[10][19:].strip())
        list_hypes.append((L*Dt, Dt, lamb, sigma, group_reg, probDrop, probZero, rep))
    return list_hypes
        

def main():
    X, ptime = read_matlab('/data/causal/golden_standards/X_Dyngen.mat')
    gene_names = [i[0] for i in (loadmat("/data/causal/golden_standards/gene_list.mat")['gene_list']).reshape(-1)]
    ptime = ptime/np.max(ptime)* 100
    #hypeparameters = parse_hyperparameters('/data/src/SINGE/default_hyperparameters.txt')
    hypeparameters = parse_hyperparameters(sys.argv[4])
    #data = np.load("/data/causal/data/simulated_dataset.npz",allow_pickle=True)
    #X,ptime,gene_names = np.asarray(data['mat'].ravel()[0].todense()),data['ptime'].reshape(-1),data['gene_names'].reshape(-1)
    sharedX = mp.RawArray(c_double, X.shape[0] * X.shape[1])
    sharedptime = mp.RawArray(c_double, len(ptime))
    sharedagg = mp.RawArray(c_double, X.shape[0] * X.shape[0])
    X_ = np.frombuffer(sharedX).reshape(*X.shape)
    ptime_ = np.frombuffer(sharedptime)
    agg_ = np.frombuffer(sharedagg).reshape(X.shape[0],X.shape[0])
    X_[:] = X
    ptime_[:] = ptime
    agg_[:] = np.zeros((X.shape[0],X.shape[0]))
    #embed()
    #L, Dt, lambdas, sigma, probDrop, probZero, rep = hypeparameters[0]
    #GLG_network(X,ptime,L,Dt,lambdas,sigma,probDrop,probZero)
    #embed()
    lock = mp.Lock()
    pool = mp.Pool(initializer=init,initargs=(lock,sharedX,sharedptime,sharedagg,X.shape[0],))
    pool.map_async(GLG_Parallel,hypeparameters)
    pool.close()
    pool.join()
    table_write(format_table(create_influence_list(agg_/np.sum(agg_),gene_names)),sys.argv[1])
    table_write(format_table(create_edge_list(agg_/np.sum(agg_),gene_names)),sys.argv[2])
    np.save(sys.argv[3],agg_)
    embed()
main()
