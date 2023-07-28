import urllib.request
import os

url = "http://images.cocodataset.org/zips/train2017.zip" # 指定链接
local_path = "./coco_path/coco2017.zip" # 指定本地位置

try:
    urllib.request.urlretrieve(url, local_path)
    print("下载完成！")
except Exception as e:
    print("下载失败：", e)