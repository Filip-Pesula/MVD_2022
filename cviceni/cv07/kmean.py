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

def getMeanError(data:Data)->float:
    groups = list()
    for i in range(data.centroids.shape[0]):
        groups.append(data.xy[data.center==i,:])
    meanError = 0
    for i,centroid in enumerate(data.centroids):
        dist = np.linalg.norm(groups[i]-centroid,axis=1)
        meanError+= np.sum(np.abs(dist))
    return meanError
def getK(data:Data,maxit:int=10,reldev = 0.0001, verbouse = 1,kmax = 20)->int:
    out = _kMean(data,1,maxit,reldev)
    meanError = list()
    meanError.append(getMeanError(out))
    meanErrorDer = [0]
    for K in range(2,data.xy.shape[0]):
        errors = [getMeanError(_kMean(data,K,maxit,reldev)) for i in range(2)]
        if verbouse>2:
            print("errors",errors)
        error = min(errors)
        if K>5 and ((error < abs(min(meanErrorDer)) and error > max(meanError)/2) or max(meanErrorDer)>0):
            break
        meanError.append(error)
        meanErrorDer.append(meanError[-1]-meanError[-2])
    K = min(range(len(meanErrorDer)), key=meanErrorDer.__getitem__)
    if verbouse>1:
        print("meanError",meanError)
        print("meanErrorDer",meanErrorDer)
    return K+1

def _kMean(data:Data,K:int,maxit:int=100,reldev = 0.01,verbouse = 1):
    xmin = np.min(data.xy[:,0])
    xmax = np.max(data.xy[:,0])
    np.random.random((data.xy.shape[0],))
    ymin = np.min(data.xy[:,1])
    ymax = np.max(data.xy[:,1])
    np.sum(data.xy[:,1])/data.xy.shape[1]
    if verbouse > 2:
        print("x:", xmin,"-", xmax)
        print("y:", ymin,"-", ymax)
    
    centroid = np.random.rand(K,data.xy.shape[1])
    centroid[:,0]=centroid[:,0]*abs(xmax-xmin)+xmin
    centroid[:,1]=centroid[:,1]*abs(ymax-ymin)+ymin
    if verbouse > 2:
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

    for iterr in range(maxit):
        if iterr>1 and np.sum(np.abs(data.centroids-centroid)) < reldev: 
            if verbouse > 0:
                print("maxitReached!!!")
            break
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
    if verbouse > 2:
        print("res",nceSum/nceCoun)
        print("resSum",np.sum(nceSum/nceCoun))

    data.centroids = centroid
    return data
def kMean(data:Data,K:int|None = None,maxit:int=100,reldev = 0.01,verbouse = 1,kmax = 20)->Data: 
    if K is None:
        K = getK(data,maxit,reldev,verbouse,kmax)
    out = _kMean(data,K,maxit,reldev,verbouse)
    return data
    
if __name__ == "__main__":
    """
    data = genBlob([(-5,5),(5,5),(5,-5),(-5,-5)],200)
    out = kMean(data,4,5)
    print(out.center)
    print(data.centroids)
    #out.show()
    out.showKmean()
    """

    blob1K = genBlob([(0,-5),(0,5)],100)
    #blob1.show()

    out1 = kMean(blob1K,verbouse = 2)
    out1.showKmean()