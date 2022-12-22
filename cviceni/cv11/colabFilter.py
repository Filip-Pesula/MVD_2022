import numpy as np


class ColabFilter:
    def __init__(self,items) -> None:
        self.items = items
        self.norm = np.sum(items,1)/np.sum(items>0,1)
        self.itemsNorm = (items.T-self.norm).T* (items>0 * 1)
        pass 
    def dist(self,a_,b_):
        a = self.itemsNorm[a_,:]
        b = self.itemsNorm[b_,:]
        return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    def closest(self):
        close = 0
        closei = (0,1)
        for i in range(np.shape(self.itemsNorm)[0]):
            for j in range(i,np.shape(self.itemsNorm)[0]):
                d = self.dist(i,j)
                if(d>close and i!=j):
                    close = d
                    closei = (i,j)
        return close,closei

if __name__ == "__main__":
    items =  np.asarray([
        [1,0,3,0,0,5,0,0,5,0,4,0],
        [0,0,5,4,0,0,4,0,0,2,1,3],
        [2,4,0,1,2,0,3,0,4,3,5,0],
        [0,2,4,0,5,0,0,4,0,0,2,0],
        [0,0,4,3,4,2,0,0,0,0,2,5],
        [1,0,3,0,3,0,0,2,0,0,4,0],
    ])
    colabf = ColabFilter(items)
    print("dist 0,0:",colabf.dist(0,0))
    print("dist 0,1:",colabf.dist(0,1))
    print("dist 0,2:",colabf.dist(0,2))

    print("closest",colabf.closest())