---
title: "用 Python 实现换脸"
date: "2020-06-26"
mathjax: true
sidebar: "right" # Enable sidebar (on the right side) per page
widgets: # Enable sidebar widgets in given order per page
  - "search"
  - "recent"
  - "taglist"
categories:
    - "Machine Learning"
tags: 
    - "Python"
    - "CV"
isCJKLanguage: true
comments: true
---

> 这篇博客的算法和代码主要是依据这篇文章：[Switching Eds: Face swapping with Python, dlib, and OpenCV](https://matthewearl.github.io/2015/07/28/switching-eds-with-python/)

## Introduction


![switchface](/switch-face-using-python/switchface.jpg)

仔细看看，觉不觉得上面的两幅图有哪里不对？
<!-- more -->
最近看了一篇很有趣的博客，简述了如何用Python调用dlib和OpenCV把两张照片中的人脸交换一。dlib是一个C++实现的机器算法的工具箱，包含了人脸识别的算法，并且提供了了Python接口。OpenCV是一个著名的图像处理的库，也可以用Python调用。在这些程序库的帮助下，我们可以只用两三百行程序就实现一个换脸的程序。

下面我们就来看看如何用Python一步步实现一个换脸的程序。[所有代码在这里](https://github.com/messcode/SwitchFace)。


## 算法简述

要实现换脸，当然首先要从图片中把人的脸部特征标记出来。这是整个算法中最复杂的部分，好在dlib提供了一个方法，可以定位图片中人的正脸，并且返回68个人脸的特征点的位置（landmark）。本文不打算介绍具体的算法，感兴趣的读者请参考[官方文档](http://dlib.net/) 。在得到了两张人脸的特征点之后，我们就可以把面部重要的特征部分例如眼睛、鼻子、嘴巴等扣出来。不妨称这两张人脸的特征部分为s，t。

接下来，就是如何实现两交换s，t。要使人脸交换之后看起来比较自然，例如把特征部分s覆盖到t上，这就要求s的轮廓要和t的轮廓尽量对齐。一个自然的想法是通过平移、旋转等基本变换，使s尽量地和t对齐。这是实现换脸的关键。

实现了人脸对齐之后，剩下的工作就很简单了。我们只需要把s通过仿射变换对齐到t上，然后覆盖住t，同样地方法也用t覆盖住s就实现了“换脸”。不过，还有一些细节上的工作需要处理。如果两人的肤色差别太大，换脸之后就会显得很突兀，所以还需要进行色差矫正使得颜色看起来更加自然。下面介绍如具体实现。

### 人脸识别

![marked_lenna](/switch-face-using-python/marked_lenna.png)

这张图就是著名的Lenna女神。这张图片经常被用作图像处理的测试图片，图上的标号就是面部的68个特征点，包括了下巴轮廓，眼睛，嘴巴，鼻子等的特征点。

dlib提供了一个默认的人脸识别器，可以通过`dlib.get_frontal_face_detector`来得到，它定位人脸所在的矩形。`dlib.shape_predictor`能够预测图片中的特征点。在这里，就是可以返回面部的68个特征点，它们包括了。你可以自己训练这个人脸识别器，不过dlib提供了训练好的数据（[下载链接](https://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2/download)）。
``` Python
# Returns the default face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
def get_landmarks(img):
    """
    Get a list  of faces' landmarks from a image.
    """
    rects = detector(img, 1) # 1 is upsampling factor.
    return [numpy.matrix([[p.x, p.y] for p in predictor(img, rect).parts()])
    		for rect in rects]
```
`get_landmarks`返回的是一个列表，列表的每个元素就是图中每个人脸的68个特征点。（图中有几张人脸，列表就包含几个元素。）

### 提取特征部分

![marked_img](/switch-face-using-python/marked_img.jpg)

在有了这68个特征点之后，如何将对应的部分提取出来呢？这里就需要用到opencv提供的工具convexHull。把特征点按照各个部分分组，鼻子，嘴巴等。然后画出包含各个组的凸包，即包含这些点的最小凸集。
``` Python
LEFT_EYE_POINTS = list(range(42, 48))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_BROW_POINTS = list(range(17, 22))
NOSE_POINTS = list(range(27, 35))
MOUTH_POINTS = list(range(48, 61))
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]

FEATHER_AMOUNT = 11

def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)

def get_face_mask(im, landmarks):
    im = numpy.zeros(im.shape[:2], dtype=numpy.float64)

    for group in OVERLAY_POINTS:
        draw_convex_hull(im,
                         landmarks[group],
                         color=1)

mask = get_face_mask(img, landmarks[0])
```
`OVERLAY_POINTS`的每个元素是一组特征点，对每组特征点用draw_convex_hull用白色填充(color = 1)， 从上面的例子是把鼻子+嘴巴作为一组，眉毛+眼睛作为一组，就可以得到下图所示的面具：
![combined_mask.jpg](/switch-face-using-python/combined_mask.jpg)



### 面部对齐

计算机中存储的图片实际上是由一个个像素点组成的，所以图片可以看成是一个二维的矩阵，每一个像素点的位置可以用它的行号和列号$(i，j)$来表示。两张人脸的特征点集合分别为{%raw%}$s = s_1,s_2...s_{68}$和$t = t_1, t_2...t_{68}${%endraw%}。

我们希望能够找到一种变换方式$T:(i,j) \to (i', j')$，使得$s_i$经过变换后离$t_i$的距离尽可能地近。即求解这样一个优化问题:
{%raw%}
$$\min \sum_{i = 1}^{68}||T(s_i) - t_i||^2$$
{%endraw%}
这个问题的求解显然是十分困难的，因为我们对$T$几乎一无所知。为了简化问题，我们考虑只通过线性变换来极小化目标函数原来的优化问题转化为：
{% raw %}
$$\begin{equation}
	\begin{split} 
    	& {\text{min}}
& &  \sum_{i = 1}^{68}||cRs_i + T - t_i||^2 \\\\
& \text{s.t.} & &  R^T R = I\\\\
	\end{split}
\end{equation}$$
{% endraw %}
其中$c$是尺度变换参数，$R$是一个旋转矩阵，$T$是二维的平移向量。这个问题就是[Orthogonal Procrustes Problem](https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem)，它可以利用SVD分解求解。利用下面的代码可以求出对应的变换矩阵。
``` Python
def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:
        sum ||c*R*p1,i + T - p2,i||^2
    is minimized.
    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)

    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    # normalization
    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = numpy.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         numpy.matrix([0., 0., 1.])])
```
我们来仔细看看这段代码做了什么:
1. 对数据进行标准化， 尺度参数就是两样本标准差的比值`s2 / s1`
2. 通过SVD分解求出旋转矩阵**R**
3. `c2.T - (s2 / s1) * R * c1.T)`就是用`points2`的中心减去`points1`的经过旋转和尺度变换后的中心，得到的结果就是平移向量。
4. 最后的返回值矩阵是一个3 * 3的矩阵：
$$
\begin{bmatrix}
x' \\\\ y' \\\\ 1
\end{bmatrix} =
\begin{bmatrix}
R & T \\\\
\mathbf{0} & 1\end{bmatrix}
\begin{bmatrix}
x\\\\ y \\\\ 1
\end{bmatrix}
$$
点的坐标采用齐次形式，**R**是旋转矩阵，T是平移向量。利用上面的`transformation_from_points`我们就可以求出把`points1`变换到`points2`的变换矩阵`M`。opencv提供的函数warpAffine，我们就可以把图片变换
```  Python
def warp_im(im, M, dshape):
	output_im = numpy.zeros(dshape, dtype=im.dtype)
	cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im
mask = get_face_mask(img, landmarks[0])
M = transformation_from_points(landmarks[1], landmarks[0])
affined_mask = warp_im(mask, M, img.shape)
warped_img = warp_im(img, M, img.shape)
```
经过变换之后，人脸基本对齐。只需要把第二列图片人脸覆盖在第左侧的人脸上就可以了：
![tranformation](/switch-face-using-python/tranformation.jpg)

### 色差矫正
两个人的由于光照、肤色等问题，在两张图拼接的边缘会出现色差。为了使得图片看起来更加自然，需要对第二张的图片的颜色进行矫正，使得它看起来和第一张图的颜色更加匹配。
``` Python
COLOUR_CORRECT_BLUR_FRAC = 0.6
LEFT_EYE_POINTS = list(range(42, 48))
RIGHT_EYE_POINTS = list(range(36, 42))
def correct_colours(im1, im2, landmarks1):
    """
    Attempt to change the colouring of im2 to match that of im1. 
    It does this by dividing im2 by a gaussian blur of im2,  and then multiplying 
    by a gaussian blur of im1.
    """
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * numpy.linalg.norm(
                              numpy.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                              numpy.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(numpy.float64) * im1_blur.astype(numpy.float64) /
                                                im2_blur.astype(numpy.float64))  
```
这个算法实际上是相当于是对`im1_blur`和`im2_blur`是两张图片高斯模糊得到的图片。`im1_blur \ im2_blur`相当于是一个对`im2`的矫正系数。得到的效果如图：


![warped_corrected_img](/switch-face-using-python/warped_corrected_img.jpg)


### Switch Faces!
做好了上面的准备工作，我们终于可以换脸了：
``` Python

def align_face(src, landmark_src, dest, landmark_dest):
    """
    Align face  in src to dest image.
    """ 
    M = transformation_from_points(landmark_dest[ALIGN_POINTS], landmark_src[ALIGN_POINTS])
    mask = get_face_mask(src, landmark_src)
    warped_mask = warp_im(mask, M, src.shape)
    combined_mask = numpy.max([get_face_mask(dest, landmark_dest), warped_mask], axis = 0)
    warped_src = warp_im(src, M, dest.shape)
    warped_corrected_src = correct_colours(dest, warped_src, landmark_dest)

    output_im = dest * (1 - combined_mask) + warped_corrected_src * combined_mask
    return output_im.astype(numpy.uint8) 

def switch_face(img_path):
    """
    Switch faces in image.
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    landmarks = get_landmarks(img)
    if len(landmarks) < 1:
        raise ImproperNumber("Faces detected is less than 2!")
    if len(landmarks) > 2:
        raise ImproperNumber("Faces detected is more than 2!")
    
    output = align_face(img, landmarks[0], img, landmarks[1])
    output = align_face(img, landmarks[1], output, landmarks[0])
    return output
```
`align_face(src, landmark_src, dest, landmark_dest)`把图片`src`中的人脸覆盖到图片`dest`中。它的具体流程:
1. 首先计算变换矩阵`M`
2. 提取特征部分`mask`并把它变换到要覆盖的位置得到`warped_mask`
3. `warped_mask`和它要覆盖的位置的特征部分取并，以保证能够完全覆盖住。
4. 色差矫正
5. 最后把`warped_im`和目标图片`dest`组合起来

换脸只需要两次调用`align_face`就可以了。最后得到的结果：


![switched_face.jpg](/switch-face-using-python/switched_face.jpg)

~~好像并没有什么不同啊~~



## 小结
dlib和openCV提供了一套方便地用来处理图像的工具，可以做很多有趣的事情。换脸只是一个例子了。openCV还可以处理视频文件，还可以自动识别视频中的人脸，然后把它们换掉，或者自动给视频中的人脸打码等等。

PS:八卦一下，lenna（就是那个戴帽子的女孩子）是图像处理常用的测试图片。这张图片实际上是从Play Boy杂志上扫描下来的，全图在[这里](http://www.guokr.com/post/38131/)。~~书上为什么不用全图啊!!!~~


---
## Reference
1. 美丽的lenna: http://www.guokr.com/post/38131/
2. https://matthewearl.github.io/2015/07/28/switching-eds-with-python/
3. https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
