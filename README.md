# ObjectTrackDetection
使用turicreate 训练目标检测 的模型.
前序文章[让我们谈谈turiCreate.聊聊iOS的机器学习](https://www.jianshu.com/p/d01872cf396b)讲到环境配置.大家回去看看就知道如何配置环境了.
目标检测这一块根据苹果的API和说明文档.踩了好多坑,才慢慢的弄出来了.
### 目标检测效果
我们的目标检测一些物体.其本质就是什么(what)和哪里(where).给定一个图像,探测器将产生实例预测.
![](http://upload-images.jianshu.io/upload_images/4676869-4199c99258771dea.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
指示这个特定的模型来检测猫和狗面部的情况。本地化的概念在这里由实例周围的边框提供。

### 数据检测
数据要求:
1.我们要的框要尽量的框住你需要识别的物体.而且尽量不要超出太多.
2.坐标原点是图片的左上角.需要框的宽、高,和中心点的坐标x、y;
3.一个图像里你需要框起的物体必须都要框住。比如我们对水果有兴趣。则图像中的水果都要框起来。否则模型可能会混淆为什么有些水果被标记为肯定，而有些水果则被标记为否定。
4.图像方向都是竖着向上的。

数据类型
```
[{'coordinates': {'height': 104, 'width': 110, 'x': 115, 'y': 216},
'label': 'ball'},
{'coordinates': {'height': 106, 'width': 110, 'x': 188, 'y': 254},
'label': 'ball'},
{'coordinates': {'height': 164, 'width': 131, 'x': 374, 'y': 169},
'label': 'cup'}]
```
是一个数组包裹着多个字典。

今天我们来识别杯子，并且要知道我的杯子在哪！
第一步:我们收集杯子图片。建议达到200张！大家一开始可以准备30 - 40张就可以了。先感受整个过程。
![](https://upload-images.jianshu.io/upload_images/4676869-ac454f05c55b9143.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

第二步：
我们既要上代码了

第一步 引入turiCreate
```
import turicreate as tc
```
读取图片数据
```
data = tc.image_analysis.load_images('cup',with_path=True)
```
我们用turiCreate展示出来看看
```
data.explore()
```
```
data
```
查看
![](https://upload-images.jianshu.io/upload_images/4676869-b8069e6fa66ff007.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

现在我们要给我们的data数据加多一列。对应每个图片中间物体的框的信息。我们通过data.explore()发现图片的顺序并不是根据文件夹的顺序排列。所以接下来我用开大招了
打开你的终端。然后cmt + c 拷贝 图片的地址，使用终端打开这张图片
![copy这里的地址](https://upload-images.jianshu.io/upload_images/4676869-11c1bd49ec8ae08e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![终端打开图片](https://upload-images.jianshu.io/upload_images/4676869-a7de8346942baf06.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

这样我们就可以依次打开图片了。
然后我们就要获取到图片中物体的框的宽高和中心值。我直接在图片上操作。首先框好一个物体的框，然后我们获取到了宽高W和H。我们在框一个框。以左上角为原点。把物体框起来。我们得到了新的宽高Wn和Hn。那么很简单。
X = Wn - W + W/2
Y = Hn - H + H/2
这样我们就得到这个框的数据了
![W和H](https://upload-images.jianshu.io/upload_images/4676869-55f86c96f2f11a6f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![Wn和Hn](https://upload-images.jianshu.io/upload_images/4676869-af532ce2c420cacc.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

数据
```
[{'coordinates': {'height': 130, 'width': 100, 'x': 142, 'y': 89},
'label': 'doraemon'}]
```
如此循环多次之后
```
array = tc.SArray([[{'coordinates': {'height': 130, 'width': 100, 'x': 142, 'y': 89},
'label': 'doraemon'}],#1

[{'coordinates': {'height': 275, 'width': 277, 'x': 474, 'y': 143},
'label': 'doraemon'}],
[{'coordinates': {'height': 127, 'width': 106, 'x': 137, 'y': 87},
'label': 'doraemon'}],
[{'coordinates': {'height': 210, 'width': 142, 'x': 144, 'y': 113},
'label': 'doraemon'}],
[{'coordinates': {'height': 424, 'width': 436, 'x': 274, 'y': 235},
'label': 'doraemon'}],
[{'coordinates': {'height': 180, 'width': 180, 'x': 246, 'y': 222},
'label': 'doraemon'}],
[{'coordinates': {'height': 232, 'width': 190, 'x': 105, 'y': 124},
'label': 'doraemon'}],
[{'coordinates': {'height': 306, 'width': 288, 'x': 330, 'y': 243},
'label': 'doraemon'}],
[{'coordinates': {'height': 169, 'width': 118, 'x': 86, 'y': 128},
'label': 'doraemon'}],
[{'coordinates': {'height': 567, 'width': 460, 'x': 403, 'y': 325},
'label': 'doraemon'}],
[{'coordinates': {'height': 227, 'width': 180, 'x': 243, 'y': 137},
'label': 'doraemon'}],
[{'coordinates': {'height': 735, 'width': 575, 'x': 288, 'y': 368},
'label': 'doraemon'}],
[{'coordinates': {'height': 395, 'width': 340, 'x': 460, 'y': 283},
'label': 'doraemon'}],
[{'coordinates': {'height': 218, 'width': 172, 'x': 230, 'y': 399},
'label': 'doraemon'}],
[{'coordinates': {'height': 271, 'width': 186, 'x': 93, 'y': 136},
'label': 'doraemon'}],
[{'coordinates': {'height': 128, 'width': 200, 'x': 154, 'y': 106},
'label': 'doraemon'}],
[{'coordinates': {'height': 87, 'width': 70, 'x': 140, 'y': 93},
'label': 'doraemon'}],
[{'coordinates': {'height': 460, 'width': 345, 'x': 448, 'y': 230},
'label': 'doraemon'}],
[{'coordinates': {'height': 390, 'width': 318, 'x': 220, 'y': 343},
'label': 'doraemon'}],
[{'coordinates': {'height': 145, 'width': 113, 'x': 157, 'y': 170},
'label': 'doraemon'}],
[{'coordinates': {'height': 275, 'width': 233, 'x': 307, 'y': 256},
'label': 'doraemon'}],
[{'coordinates': {'height': 200, 'width': 165, 'x': 100, 'y': 130},
'label': 'doraemon'}],
[{'coordinates': {'height': 172, 'width': 147, 'x': 232, 'y': 98},
'label': 'doraemon'}],
[{'coordinates': {'height': 160, 'width': 104, 'x': 89, 'y': 157},
'label': 'doraemon'}],
[{'coordinates': {'height': 208, 'width': 144, 'x': 148, 'y': 110},
'label': 'doraemon'}],
[{'coordinates': {'height': 446, 'width': 395, 'x': 213, 'y': 324},
'label': 'doraemon'}],
[{'coordinates': {'height': 200, 'width': 145, 'x': 98, 'y': 100},
'label': 'doraemon'}],
[{'coordinates': {'height': 816, 'width': 979, 'x': 490, 'y': 408},
'label': 'doraemon'}],
[{'coordinates': {'height': 127, 'width': 90, 'x': 214, 'y': 76},
'label': 'doraemon'}],
[{'coordinates': {'height': 262, 'width': 195, 'x': 109, 'y': 216},
'label': 'doraemon'}],
[{'coordinates': {'height': 189, 'width': 230, 'x': 125, 'y': 104},
'label': 'doraemon'}],
[{'coordinates': {'height': 775, 'width': 650, 'x': 425, 'y': 420},
'label': 'doraemon'}],
[{'coordinates': {'height': 570, 'width': 463, 'x': 405, 'y': 322},
'label': 'doraemon'}],
[{'coordinates': {'height': 244, 'width': 203, 'x': 233, 'y': 153},
'label': 'doraemon'}],
[{'coordinates': {'height': 160, 'width': 106, 'x': 91, 'y': 155},
'label': 'doraemon'}],
[{'coordinates': {'height': 140, 'width': 206, 'x': 169, 'y': 90},
'label': 'doraemon'}],
[{'coordinates': {'height': 190, 'width': 190, 'x': 236, 'y': 153},
'label': 'doraemon'}],
[{'coordinates': {'height': 136, 'width': 123, 'x': 136, 'y': 94},
'label': 'doraemon'}],
[{'coordinates': {'height': 200, 'width': 200, 'x': 445, 'y': 300},
'label': 'doraemon'}],
[{'coordinates': {'height': 275, 'width': 247, 'x': 124, 'y': 138},
'label': 'doraemon'}],
[{'coordinates': {'height': 190, 'width': 290, 'x': 150, 'y': 148},
'label': 'doraemon'}]])
```
则得到了我们的数据了
我们要把框的信息加到数据的新列中去
```
data = data.add_column(array,column_name='annotations')
```
我们在打印一下数据
```
data
```
![](https://upload-images.jianshu.io/upload_images/4676869-9d0c5e02eb307e13.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
发现我们成功了。
注意的时候data数据里每个列的行数都必须要一致，否则就添加不进去。

到这里我们数据准备好了。是不是就可以开始创建模型了呢！按照apple的turiCreate API 是开始创建模型了。但是很不辛的是，你创建模型是失败的。为什么？
是因为图片不支持4通道的。https://github.com/apple/turicreate/issues/312 这里有苹果官方关于这方面的解释
那我们要怎么办呢最快的方法是
```
data['image'] = data['image'].apply(lambda image: tc.image_analysis.resize(image,image.width,image.height,3))
```
就可以把图片转换为3通道的了。
终于到了我们创建模型的时候了
```
model = tc.object_detector.create(data, annotations='annotations', feature='image', model='darknet-yolo', classes=None, max_iterations=0, verbose=True)
```
这个时候你就只需要等待了。不错等待。这个时间最少都是3个钟头以上。当然有更快的方法是 使用GPU。然而我们公司的inter的，turiCreate支持的CUDA。所以我没有使用过。后续我在尝试一下。

号外号外
如果你按照上面的步骤发现创建模型还是有问题的。那就可能是你的图片有问题，具体是什么问题我也还不知道。所以我们在发现到这个时候创建模型还是不可以的时候。我们就排除一下那些图片是不可以的。方法非常暴力就是10张 10张添加，依次发现哪张图片是有问题的。
最后我们是导出模型
```
model.export_coreml('objectTracking.mlmodel')
```
![目标检测模型](https://upload-images.jianshu.io/upload_images/4676869-c917c157aede9c2f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

那么我们只需要导入到Xcode就可以使用了.

[Xcode demo](https://www.jianshu.com/p/b7993fc032da)
