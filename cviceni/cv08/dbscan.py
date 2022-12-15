from DataClass import Data
import DataClass
import numpy as np
import queue
import random
class Node():
    
    def __init__(self, i:int, parent) -> None:
        self.i = i
        self.chidCount = 0
        self.parent = parent
        if(self.parent!=None):
            self.parent.chidCount+=1
        pass
    def root(self)->any:
        if self.parent is not None:
            #print("jump: ",self.parent.i)
            return self.parent.root()
        else:
            return self
    def up(self):
        """
        push Node directly below root
        """
        parent = self.parent.parent
        while parent is not None:
            self.parent = parent
            parent = self.parent.parent
    def pop(self)->bool:
        if self.leaf():
            self.parent.chidCount -= 1
            self.parent = None
            return True
        else:
            return False
    def move(self,newRoot)->bool:
        if self.leaf():
            if self.parent is not None:
                self.parent.chidCount -= 1
            self.parent = newRoot
            newRoot.chidCount += 1
            self.up()
            return True
        else:
            return False
    def leaf(self)->bool:
        return self.chidCount==0

def closest(point:int, data:Data,pointsNodes:list[Node],eps:float)-> list[int]:
    inRange = list()
    for i in range(data.xy.shape[0]):
        dist = np.linalg.norm(data.xy[i,:] - data.xy[point,:])
        if dist< eps:
            inRange.append(i)
    inRange.remove(point)
    for cloasePoint in inRange:
                if (pointsNodes[cloasePoint].root().i != -1 and
                pointsNodes[cloasePoint].root().i != pointsNodes[cloasePoint].root().i):
                    inRange.remove(cloasePoint) 
    return inRange

def addToQue(que, closePoints,passed):
    for point in closePoints:
            if point not in que.queue and point not in passed:
                que.put(point)
                passed.append(point)
            else:
                #print("parent move Error")
                pass


def scan(data:Data,eps:float = 1,MinPts:int = 3, maxit = 500):
    groups = [-1]
    emptyRoot = Node(-1,None)
    pointsNodes = list()
    for i in range(data.xy.shape[0]):
        pointsNodes.append(Node(i,emptyRoot))
    empty = [*range(data.xy.shape[0])]
    while len(empty)>0:
        select = empty[0]
        empty.remove(select)
        #chceck if valid for starting group
        closePoints = closest(select,data,pointsNodes,eps)
        if len(closePoints)<MinPts:
            continue
        pointsNodes[select].pop()
        groups.append(select)
        que = queue.Queue()
        passed = [select]
        addToQue(que,closePoints,passed)
        #prosess group
        while not que.empty():
            eval = que.get()
            passed.append(eval)
            closePoints = closest(eval,data,pointsNodes,eps)
            #přímo dosažitelný
            if len(closePoints)>=MinPts:
                if pointsNodes[eval].leaf():
                    pointsNodes[eval].move(pointsNodes[select])
                    if eval in empty:
                        empty.remove(eval)
                addToQue(que,closePoints,passed)
            #nepřímo dosažitelný
            else:
                if pointsNodes[eval].leaf():
                    pointsNodes[eval].move(pointsNodes[select])
                    if eval in empty:
                        empty.remove(eval)
    for i in range(len(data.l)):
        data.l[i] = groups.index(pointsNodes[i].root().i)
    print("groups",groups)
    return data
                

    pass

if __name__ == "__main__":
    data = DataClass.genMonns([(0,0.5),(0.5,0)],200)
    scan(data,0.3,12)
    data.show()