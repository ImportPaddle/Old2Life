import paddle

x = paddle.load('x.pth')
print(x)

for i,d in enumerate(x):
    print(i)

print(2)