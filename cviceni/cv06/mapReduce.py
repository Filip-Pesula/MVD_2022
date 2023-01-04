import functools
from collections import defaultdict
import csv
import multiprocessing

def myMap(data):
    i,title = data
    mapDataRext = dict()
    for word in title.split():
        if not word in mapDataRext:
            mapDataRext[word] = (f"D{i}",1)
        else:
            mapDataRext[word] = (mapDataRext[word][0], mapDataRext[word][1]+1)
            mapDataRext.items
    return list(mapDataRext.items())
def reduce(mappedData:list):
    kv = [item for sublist in mappedData for item in sublist]
    d = defaultdict(list)
    for key, value in kv:
        d[key].append(value)
    return d


def load(inf, pooled = False):
    global pbar
    titlesList = list()
    print("read")
    with open(inf,"r",encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)
        for i,line in enumerate(reader):
            titlesList.append((i,line[4]))
    print("map")
    if pooled:
        with multiprocessing.Pool(processes=4) as pool:
            maplsit = list(pool.map(myMap,titlesList,chunksize=10))
    else:
        maplsit = list(map(myMap,titlesList))
    print("reduce")
    return titlesList,reduce(maplsit)
if __name__ == "__main__":
    titles, inverted_index = load("articlesLEMMA.csv",pooled = True)
