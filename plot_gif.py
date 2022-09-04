# -*- coding: utf-8 -*-
import imageio
import os
'''
# 只支持png格式，需要先命名排序好(默认按照字母序排列)
# source(字符串)：素材图片路径，生成的gif也保存在该路径
# gifname(字符串)：生成的gif的文件名，命名时带后缀如：'1.gif'
# time(数字)：生成的gif每一帧的时间间隔，单位（s）
'''
def png2gif(source, gifname, time):
    os.chdir(source) # os.chdir()：改变当前工作目录到指定的路径
    file_list = os.listdir() # os.listdir()：文件夹中的文件/文件夹的名字列表
    frames = [] #读入缓冲区
    for png in file_list:
        print(1)
        frames.append(imageio.imread(png))
    imageio.mimsave(gifname, frames, 'GIF', duration=time)
address = "./simulations/"
png2gif(address, 'result.gif', 0.1)