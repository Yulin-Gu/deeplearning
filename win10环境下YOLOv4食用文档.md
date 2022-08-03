# win10下YOLOv4食用文档 #



## 一、配置YOLOv4运行环境 ##

#### 1. 配置环境 ####

* 先是参考[这个知乎帖子](https://zhuanlan.zhihu.com/p/45845454) 第五点测试darknet前面的所有步骤

* ps: 对于此篇帖子里说的vs不能安装过高的版本, 我是没有因为这个而安装失败的, 因为我之前已经安装过`vs2019`, 懒得重新安装低版本的, 所以没有听这个作者的“劝告”斗胆用高版本的vs2019去配置了, 结果好像也没有啥问题出现. (darknet确实是在vs2015上开发的, 但是当我用vs2019打开`darknet.sln` 的时候, vs2019会提示我此文件是在低版本的vs上开发的, 问我是否需要重新生成解决方案, 然后我点了确定, 如果不重新生成解决方案的话, 之后生成darknet就会报错)



#### 2. 参考上面教程配置下来遇到的新的问题以及解决方案:  ####

1. 想要生成`darknet.exe` 的时候`vs2019`报错: **找不到version.cpp问题**

   凡是这类找不到`opencv` 里面的某个文件的问题, 基本上都是vs配置`opencv`出错了: **库目录或者包含目录或者链接没有配置好.**

   

   这是为什么呢? 

   不一定是你刚开始没有按照教程配置好——我发现一个比较迷幻的问题是: 即使我反复检查过很确定在vs里配好了`opencv` , 并且成功生成了darknet.exe, 然后, 我闲着无聊, 关闭了`darknet.sln `重启了一遍想要再生成一次darknet, 此前配置好的`opencv `的那些包含目录, 库目录和链接, 可能就会没了(对, 是可能, 因为我试了好多次. 它有时候可以保存到你刚才的配置, 有时候重新打开就跟配置之前一样, 然后又得重新配置一遍) . 具体原因不清楚, 所以如果遇到此类问题, 打开属性页检查一下`OpenCV` 是否配置好了.

   

2. 生成`darknet.exe `的时候`vs2019 `报错: **Error MSB3721**

   出现次错误的原因是因为电脑显卡算力不够, 在`darknet.vcxproj `文档里修改算力: 用记事本 / `notepad ` 方式打开该文件, 检索compute关键词, 然后找到:`  line107`和 `line158`, 分别把这两行里的第二个`computer_` 和第二个 `sm_`后面的数字都改为52. 当然, 你也可以选择改为你电脑算力*10的数字, 比如我的电脑算力是6.1, 也可以改为61.

   

#### 3. 测试darknet ####

* 做完以上的步骤之后, 终于成功生成darknet.exe啦, (该程序放在了`build\darknet\x64`目录下). 

* 到[这个网站](https://drive.google.com/file/d/1hSrVqiEbuVewEIVU4cqG-SqJFUNTYmIC/view)上下载`yolov4.cfg`文件,  [这个网站](https://drive.google.com/file/d/1L-SO373Udc9tPz5yLkgti5IAXFboVhUt/view)下载`yolov4.weithgts`文件. 下载完成之后把这两个文件放进`build\darknet\×64`目录下. 
* 在`build\darknet\×64` 目录下运行`darknet detect yolov4.cfg yolov4.weights dog.jpg`.  结果`opencv` 报错: **由于找不到opencv_highgui2411d.dll,无法继续执行代码。重新安装程序可能会解决此问题**. 解决方案是把目录`C:\OpenCV\opencv\build\x64\vc14\bin` 下的所有`dll` 文件复制到`C:\Windows\System32`目录下面,
* 重新运行`darknet detect yolov4.cfg yolov4.weights dog.jpg`, 运行完成之后弹出一张带有bounding box的狗的图片即说明darknet总算安装成功了. 



## 二、准备数据集 ##

#### 用`vott` 给自己的数据集打标记.  ####

* 使用的数据集是老师给的`gun_sword`的数据集,`vott`用的是和老师同样的1.72的版本, 到 `vott`的官网下载, 按照指引一步步安装即可, 使用教程参照老师上课的演示. 

* 标记完成之后导出数据, 把`vott` 导出的`output\data`文件夹里的`obj`文件夹、 `obj.names`文件、`obj.data`文件、`test.txt`文件`和train.txt`文件全都放进`build\darknet\×64\data\`文件夹里. 



## 三、调模型, 训练数据集 ##

1. 在[这里](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29)下载`yolov4-tiny29.conv.29` 文件, 下载完成之后把这个文件放到`build\darknet\×64\`目录下
2. 在`build\darknet\×64\`目录下新建一个文件`yolo-tiny-obj.cfg`,  把[这个文件](https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4-tiny.cfg)里的所有内容复制到`yolo-tiny-obj.cfg `中. 
3. 对`yolo-tiny-obj.cfg ` 文件做以下修改: 
   * 更改`batch`为`batch = 64`
   * 更改`subvision`为`subvision = 16`
   * 更改`max_batches`为`max_batches = 6000`, 更改`steps`为`steps = 4800, 5400`
   * 更改network size为 `width = 416, height = 416`
   * 更改`line220` 和`line269的` `classes` 为`classes = 2`
   * 更改`line212` 和 `line263` 的`filters = 255` 为 `filters = 21`

4. 在`build\darknet\×64`目录下运行命令`darknet.exe detector train data/obj.data yolov4-tiny-obj.cfg yolov4-tiny.conv.29 -map`

5. 训练完成之后得到结果图如下: 

   

![chart_yolov4-tiny-obj](https://i.loli.net/2021/05/25/Xafld5CIVnU3Ret.png)



PS: 我最开始下载的是`yolov4.conv.137` , 本来是想用YOLOv4的137层进行训练的, 但是运行的时候出现报错: `Error: cuDNN isn't found FWD algo for convolution`  . 然后在`github`查了`issue`发现是因为内存不够, 所以改为用29层进行训练. 



## 四、验证数据集和计算map ##

1. 到[这个网站](https://github.com/Cartucho/mAP)下载这个项目的代码
2. 把验证集中的测试图片放到`map\input\images-optional`目录下
3. 把用`vott`输出的所有测试图片对应的`.txt`文件放到`map\input\ground-truth`目录下
4. 把`map\scripts\extra`目录下的`class_list.txt`里面的所有内容删除, 第一行输入gun, 第二行输入sword, 然后保存文件. 
5. 把`yolo-tiny-obj.cfg` 文件里面的`subvision`和`batch` 改为1
6. 把训练生成的`×64\backup\`文件夹下的`yolov4-tiny-obj_6000.weights` 文件复制到`×64`目录下
7. 在`build\darknet\×64` 目录下运行`darknet.exe detector test data/obj.data yolov4-tiny-obj.cfg yolov4-tiny-obj_6000.weights -dont_show -ext_output < data/test.txt > result.txt -thresh 0.25` 
8. 运行完成之后在`build\darknet\×64` 目录下得到一个`result.txt` 文件, 把这个文件复制粘贴到`map\input\detection_results` 目录下
9. 打开`map\scripts\extra`目录下的`convert_dr_yolo.py` 文件, 把里面的代码全部删掉, 换为以下代码, 然后保存文件, 并且运行该文件. 运行完成之后把上述第八个步骤得到的`result.txt`文件删除掉. 

```python
import os
import re

# make sure that the cwd() in the beginning is the location of the python script (so that every path makes sense)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

IN_FILE = 'result.txt'

# change directory to the one with the files to be changed
parent_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
parent_path = os.path.abspath(os.path.join(parent_path, os.pardir))
DR_PATH = os.path.join(parent_path, 'input','detection-results')
#print(DR_PATH)
os.chdir(DR_PATH)

SEPARATOR_KEY = 'Enter Image Path:'
IMG_FORMAT = '.jpg'
result_format = '%'

# outfile = None
fo = open(os.path.join(DR_PATH, 'result.txt'), "r")
alllines = fo.readlines()  #依次读取每行  
   # alllines = alllines.strip()    #去掉每行头尾空白  
# 关闭文件
fo.close()
for line in alllines:
    #if SEPARATOR_KEY in line:
    if IMG_FORMAT in line:
    #    if IMG_FORMAT not in line:
    #        break
        # get text between two substrings (SEPARATOR_KEY and IMG_FORMAT)
        #image_path = re.search(SEPARATOR_KEY + '(.*)' + IMG_FORMAT, line)
        # get the image name (the final component of a image_path)
        # e.g., from 'data/horses_1' to 'horses_1'
        #image_name = os.path.basename(image_path.group(1))
        image_path = (line.split(':', 1))[0]
        image_name = (image_path.split('/',2))[2]
        image_name = (image_name.split('.',1))[0]
        
        
        # close the previous file
        #if outfile is not None:
        #    outfile.close()
        # open a new file
        
    # elif outfile is not None:
    elif result_format in line:
        # split line on first occurrence of the character ':' and '%'
        outfile = open(os.path.join(DR_PATH, image_name + '.txt'), 'w')
        class_name, info = line.split(':', 1)
        confidence, bbox = info.split('%', 1)
        # get all the coordinates of the bounding box
        bbox = bbox.replace(')','') # remove the character ')'
        # go through each of the parts of the string and check if it is a digit
        left, top, width, height = [int(s) for s in bbox.split() if s.lstrip('-').isdigit()]
        right = left + width
        bottom = top + height
        outfile.write("{} {} {} {} {} {}\n".format(class_name, float(confidence)/100, left, top, right, bottom))
        outfile.close()
        
```



10. 进入`map\scripts\extra\`目录, 打开其中的`convert_gt_yolo.py`文件, 把其中`line62` 和`line58` 的图片路径改为`images-optional`, 然后运行该文件.

11. 进入`map\`目录运行`main.py` 文件, 得到计算出来的map结果如下:

    

    ![mAP](https://i.loli.net/2021/05/25/rgBiIfeFjDmoHNn.png)