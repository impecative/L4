import glob
import os

graphs = glob.glob("*.png")

appendfname = "_50mm_10000_combinations.png"



for graph in graphs:
    name = graph.split(".")[0]
    os.rename(graph, name+appendfname)

