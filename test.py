# -*- coding: cp936 -*-

import cv2
import sys
import time
some_ops = lambda x, y: x + y + x*y + x**y
some_ops(2, 3)  # 2 + 3 + 2*3 + 2^3 = 19

print(some_ops(2,3))

print('===============\n')

#从10倒数到0
def countDown(x):
    while x >= 0:
        yield x
        x -= 1

for i in countDown(10):
    print(i)

print("print fibonacci data === \n")
# 打印小于100的斐波那契数
def fibonacci(n):
    a = 0
    b = 1
    while b < n:
        yield b
        a, b = b, a + b


for x in fibonacci(100):
    print(x)


print("=============\n")
A = reduce(lambda x, y: x * y, [1, 2, 3, 4])    # ((1+2)+3)+4=10
print(A)
print("=================\n")

cv2.namedWindow("Test")

cap = cv2.VideoCapture("I:\\TangJY\\MOD-H7-recordData\\20190103\\1.mp4")

while 1:
    ret, frame = cap.read() #读取
    cv2.imshow("Test", frame) #显示
    if cv2.waitKey(40) & 0xff == ord('q'): #按q退出
        break

cap.release();

# img = cv2.imread("E:\\matlab_project\\libcbdetect\\data\\01.png", 1)
# cv2.imshow("Test", img)
# cv2.waitKey(0)
cv2.destroyAllWindows()

time.sleep(5)
