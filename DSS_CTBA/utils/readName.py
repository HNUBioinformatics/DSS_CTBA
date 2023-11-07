# 导入os模块
import os

# path定义要获取的文件名称的目录（C盘除外）
# path = "E:\工作表格\数据统计\网站日志\日志"
path = "D:/焦晓琳/04-数据/数据集2"

# os.listdir()方法获取文件夹名字，返回数组
file_name_list = os.listdir(path)
print(file_name_list)
# 转为转为字符串
file_name = str(file_name_list)

# replace替换"["、"]"、" "、"'"
file_name = file_name.replace("[", "").replace("]", "").replace("'", "").replace(",", "\n").replace(" ", "")
# 创建并打开文件list.txt
# f = open('./results/result.txt', 'a')

# 将文件下名称写入到"文件list.txt"
# f.write(file_name)

# print(file_name)