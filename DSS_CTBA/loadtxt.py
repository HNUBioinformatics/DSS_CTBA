"""
# -*- coding: utf-8 -*-
# @Author :  JJ
# @File : loadtxt.py
# @Time : 2023/4/27 10:45
# code is far away from bugs with the god animal protecting
#         ┌─┐       ┌─┐
#      ┌──┘ ┴───────┘ ┴──┐
#      │                 │
#      │       ───       │
#      │  ─┬┘       └┬─  │
#      │                 │
#      │       ─┴─       │
#      │                 │
#      └───┐         ┌───┘
#          │         │
#          │         │
#          │         │
#          │         └──────────────┐
#          │                        │
#          │                        ├─┐
#          │                        ┌─┘
#          │                        │
#          └─┐  ┐  ┌───────┬──┐  ┌──┘
#            │ ─┤ ─┤       │ ─┤ ─┤
#            └──┴──┘       └──┴──┘
"""



with open('./results/result.txt','r') as f:
	lines = f.readlines()
	
# print(lines)
lines = [i[:-1] for i in lines]   # 去掉每一行末尾的换行符

result1_all = []
result2_all = []
result3_all = []

for line in lines:
	# print(line.split(' '))
	result1_all.append(float(line.split(' ')[1]))
	result2_all.append(float(line.split(' ')[2]))
	result3_all.append(float(line.split(' ')[3]))
	
print(sum(result1_all) / len(result1_all))
print(sum(result2_all) / len(result2_all))
print(sum(result3_all) / len(result3_all))