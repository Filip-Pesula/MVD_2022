import numpy as np
import plotly.express as px

from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons

class Data:
    centroids:np.ndarray
    def __init__(self,xy:np.ndarray,l:np.ndarray) -> None:
        if type(xy)!= np.ndarray:
            raise TypeError("xy: "+"expected np.ndarray - given: "+str(type(xy)))
        self.xy:np.ndarray = xy
        if type(l)!= np.ndarray:
            raise TypeError("l: "+"expected np.ndarray - given: "+str(type(l)))
        self.l:np.ndarray = l
        self.center =  np.zeros(l.shape,dtype=int)
        self.datak:np.ndarray = np.zeros(self.xy.shape)
        pass
    def show(self)->None:
        cols = [str(i) for i in self.l]
        fig = px.scatter(self.xy, x=self.xy[:,0], y=self.xy[:,1],color=cols, text=self.l)
        fig.show()
    def showKmean(self)->None:
        cols = [str(i) for i in self.center]
        fig = px.scatter(self.xy, x=self.xy[:,0], y=self.xy[:,1],color=cols, text=self.center)
        ce = range(self.centroids.shape[0])
        colsCentroid = [str(i) for i in ce]
        fig.add_traces(
            px.scatter(self.centroids, x=self.centroids[:,0], y=self.centroids[:,1],color=colsCentroid, text=ce).update_traces(marker_size=10, marker_color="yellow").data
        )
        fig.show()


def genBlob(centers:list[tuple],num,dim = 2):
    blobs = make_blobs(num,dim,centers = centers)
    return Data(blobs[0],blobs[1])

def genMonns(centers:list[tuple],num,dim = 2):
    moons = make_moons(num, noise=0.1, random_state=42)
    return Data(moons[0],moons[1])

def genRing(centers:list[tuple],num,dim = 2):
    centr = [(i[0],i[1]) for i  in centers]
    r = [i[1] for i  in centers]
    excav = [i[2] for i  in centers]
    circle = make_circles(n_samples=num,noise=0.1,  factor=0.1)
    return Data(circle[0],circle[1])


    
class Param:
    def __init__(self,x,y,r,excav,xmask,ymask) -> None:
        self.x
        self.y
        self.r
        self.excav
        self.xmask
        self.ymask
        pass