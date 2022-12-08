import numpy as np
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
import plotly.express as px

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


def kMean(data:Data,K:int,numit:int=10,numtest:int = 100)->Data:
    xmin = np.min(data.xy[:,0])
    xmax = np.max(data.xy[:,0])
    np.random.random((data.xy.shape[0],))
    ymin = np.min(data.xy[:,1])
    ymax = np.max(data.xy[:,1])
    np.sum(data.xy[:,1])/data.xy.shape[1]
    print("x:", xmin,"-", xmax)
    print("y:", ymin,"-", ymax)
    
    centroid = np.random.rand(K,data.xy.shape[1])
    centroid[:,0]=centroid[:,0]*abs(xmax-xmin)+xmin
    centroid[:,1]=centroid[:,1]*abs(ymax-ymin)+ymin
    print("centroid",centroid)

    data.centroids = centroid

    for i in range(data.xy.shape[0]):
        minDist = float('inf')
        centroidId = 0
        for j in range(centroid.shape[0]):
            dist = np.linalg.norm(data.xy[i] - centroid[j,:])
            if dist < minDist:
                minDist = dist
                centroidId = j

        data.center[i] = centroidId

    for iterr in range(numit):
        #print("it:",iterr)
        #move centroid to center of assgned points
        nceSum = np.zeros(centroid.shape)
        nceCoun = np.zeros((centroid.shape[0],1))
        for i in range(data.xy.shape[0]):
            j = data.center[i]
            nceSum[j]+=data.xy[i]
            nceCoun[j]+=1
        centroid = nceSum/nceCoun
        #print("centroid",centroid)
        #move centroid to center of assgned points
        for i in range(data.xy.shape[0]):
            minDist = float('inf')
            centroidId = 0
            for j in range(centroid.shape[0]):
                dist = np.linalg.norm(data.xy[i] - centroid[j,:])
                if dist < minDist:
                    minDist = dist
                    centroidId = j

            data.center[i] = centroidId

    nceSum = np.zeros((centroid.shape[0],1))
    nceCoun = np.zeros((centroid.shape[0],1))
    for i in range(data.xy.shape[0]):
        j = data.center[i]
        dist = np.linalg.norm(data.xy[i] - centroid[j,:])
        nceSum[j]+=dist
        nceCoun[j]+=1
    print("res",nceSum/nceCoun)
    print("resSum",np.sum(nceSum/nceCoun))

    data.centroids = centroid
    return data
    
if __name__ == "__main__":
    data = genBlob([(-5,5),(5,5),(5,-5),(-5,-5)],200)
    out = kMean(data,4,5)
    print(out.center)
    print(data.centroids)
    #out.show()
    out.showKmean()