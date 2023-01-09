import numpy as np
import cv2 as cv
import random


def ngc(pth1, pth2):
    """
        采用 Normalized Greyscale Correlation 计算两个 patch 之间的相似度
    """
    pth1 = np.array(pth1)
    pth2 = np.array(pth2)
    
    dif1 = pth1 - np.mean(pth1)
    dif2 = pth2 - np.mean(pth2)
    
    numer = np.sum(np.multiply(dif1, dif2))
    denom = np.sqrt(np.multiply(np.sum(dif1**2), np.sum(dif2**2)))
    # print(f"ngc -> numer: {numer}, denom: {denom}")

    return numer / denom


def similarity(cltr1, cltr2):
    """
        计算两个 cluster 之间的相似度
    """
    cltr1 = np.array(cltr1)
    cltr2 = np.array(cltr2)
    ngc_list = []
    for pth1 in cltr1:
        for pth2 in cltr2:
            ngc_list.append(ngc(pth1, pth2))

    return np.sum(ngc_list) / len(ngc_list)


def cluster(cltr_list, th=0.8):
    """
        平均相似度大于阈值就合并，将得到聚类后的 cluster
    """
    cltr_list = list(cltr_list)
    class_cltr = []     # 存放最终的聚类结果
    while len(cltr_list) != 0:
        index = 0   # 类似于指针 指向下一个元素
        new_cltr = []   # 存放新类
        pop_one = cltr_list.pop(index)  # 弹出最左边的元素
        new_cltr.append(list(pop_one))    # 将弹出元素添加到新类
        while len(cltr_list)!=0:
            if similarity(new_cltr, [cltr_list[index]])>th:
                pop_two = cltr_list.pop(index)  # 弹出满足条件的元素
                new_cltr.append(list(pop_two))    # 将弹出的元素添加到新类
                index -= 1
            index += 1  # 不满足情况 指针右移

            if index >= len(cltr_list):     # 防止溢出
                break
        class_cltr.append(np.array(new_cltr))

    return class_cltr


if __name__ == "__main__":
    patch1 = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
    patch2 = np.array([[3, 12, 3],[6, 15, 6],[9, 18, 9]])
    patch3 = np.array([[63, 121, 223],[68, 135, 226],[96, 186, 229]])
    patch4 = np.array([[3, 127, 3],[6, 125, 6],[9, 165, 9]])

    # 数据集准备
    data = []
    for i in range(4):
        data.append(patch1.flatten())
        data.append(patch2.flatten())
    for i in range(6):
        data.append(patch3.flatten())
        data.append(patch4.flatten())

    print("data shape: ", np.array(data).shape)

    # 聚合聚类
    class_cluster = cluster(data)
    print(f"class number: {len(class_cluster)}")

    for i in range(len(class_cluster)):
        print(f"class {i+1} shape: {class_cluster[i].shape}")

