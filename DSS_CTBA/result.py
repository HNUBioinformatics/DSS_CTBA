import torch
pthfile = r'D:\焦晓琳\00-代码\DSAC-main\dsacepoch6.pth'
net = torch.load(pthfile)
print(net)
