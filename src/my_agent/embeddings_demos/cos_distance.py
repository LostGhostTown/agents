import numpy as np
def cosin_distance(x,y):
    """计算两个向量的余弦距离，即相似度"""
    return np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))