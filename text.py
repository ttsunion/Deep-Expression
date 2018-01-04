import os
from parameters import params as pm
string = [' ', '%', '而', '对', '面', '楼', '市', '成', '交', '抑', '制', '作', '用',
'最', '大', '的', '限', '购', '也', '成', '为', '地', '方', '政', '府', '眼', '中', 
'钉',  '自', '六', '月', '底', '呼', '和', '浩', '特', '率', '先', '宣', '布', '取',
'消', '后', '各', '便', '纷', '跟', '进']
str2num = {}
num2str = {}
for i in range(len(string)):
    str2num[string[i]] = i 
    num2str[i] = string[i]
text = '而对面楼市成交抑制作用最大限购%'
def text2label(text):
    label = [str2num[text[i]] for i in range(len(text))]
    if len(label) <= pm.Tx:
        label += [0] * (pm.Tx - len(label))
    else:
        label = label[:pm.Tx]
    return label
