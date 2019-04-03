# coding=utf-8
import re
import sys
import time

readFile = raw_input("read file path name : ")
writeFile = raw_input("write file path name : ")

# readFile = r"I:\TangJY\MOD-H7-recordData\20190103\2019-01-03can.log"
# writeFile = r"I:\TangJY\MOD-H7-recordData\20190103\test5.log"

f = open(readFile)
lines = f.readlines()
numLines = len(lines)
rowsGet = []
text = ''
for i in range(numLines):
    line = lines[i]
    # if line:
    #     str = line[15:]
    #     text = text + str
    #     print str
    # else:
    #     continue
    #---------------------------------------
    p2 = r"(\[.*?\])"
    p3 = r"(?<=\[).+?(?=\])"
    pattern1 = re.compile(p3)
    matcher1 = re.search(pattern1, line)  # 同样是查询
    ret = matcher1.group(0)
    print ret
    if ret:
        # str = line.strip(ret)

        #-----------------------------------
        str1 = re.sub('[\[\] ]', '', line)
        str = str1.strip(ret)

        print (str)
        str.lstrip()
        text = text + str
        # rowsGet.append(i)
        print str
        time.sleep(1)
    else:
        print "No match"
        continue

with open(writeFile,'w') as f:
    f.write(text)

print "=================================\n"

key = r"[15-24-07.132] 6081632 -537.599976 5.050000 3.850000 39 172 1"#这是源文本
p1 = r".*?(\[.*?\]) "#这是我们写的正则表达式
p2 = r"(\[.*?\]) "
pattern1 = re.compile(p2)#同样是编译
matcher1 = re.search(pattern1,key)#同样是查询

ret = matcher1.group(0)
# re.sub("\["+ret+"\]", '', key)
# key.replace(ret,"")
str = key.strip(ret)
str.lstrip()

if ret:
    print "match result --->",str
else:
    print "No match"