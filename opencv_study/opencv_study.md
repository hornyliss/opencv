# opencv_study

## 课程学习

OpenCV 是一个广泛用于计算机视觉任务的库，结合 C++ 能执行许多图像处理和计算机视觉的功能。以下是 C++ OpenCV 的核心知识点，涵盖了从基本操作到高级应用的各个方面：

### 1. 基本概念与安装配置
- **OpenCV 概述**：OpenCV 是什么，支持的功能，版本选择。
- **安装 OpenCV**：在 Windows/Linux/macOS 上安装 OpenCV，CMake 配置，配置编译环境。
- **集成 OpenCV 到项目**：如何在 C++ 项目中链接 OpenCV，编译器设置（如 Visual Studio、GCC）。

对于opencv在vs 2022 上面的环境配置已经讲解过了。





### 2. 基本操作
- **Mat 类**：OpenCV 中的图像数据结构，如何创建、初始化和操作 Mat 对象。Mat类是OpenCV中图像的容器，它是一个矩阵，存储图像的像素值，创建方法如下：

```c++
cv::Mat img; // 创建一个Mat对象
```





- **读取和保存图像**：`imread()`，`imwrite()`，图像的加载、显示和保存。

1. 图像的加载

```c++
cv::Mat img = cv::imread("path");  // 更换为自己的图像路径即可
```



2. 图像的显示

```c++
cv::imshow("name", img);  // 指定要进行显示的照片
```



3. 图像的保存

`imwrite(path, img)`

- `path`：图像的保存路径；
- `img`：要保存图像的名称（变量名称）。

```c++
cv::imwrite("path", img);  // 保存图像的路径和要保存的图像
```



示例：

```c++
cv::Mat img;  // 创建Mat对象
img = cv::imread("path");  // 读取图像
cv::imwrite("path", img);  // 另存图像
```



- **视频读取和保存**：`VideoCapture()`，`VideoWriter()`，从视频文件或相机读取视频帧，保存视频。

```c++
cv::VideoCapture cap(0);  // 打开摄像头
cv::Mat frame;  // 创建一个Mat对象
cap >> frame;
cv::VideoWriter writer("", cv::VideoWriter::fourcc('M','J', 'P', 'G'), 30, frame.size());
writer.write(frame);
```





- **图像显示**：`imshow()`，窗口操作，`waitKey()`。

```c++
cv::imshow("window", img);
cv::waitKey(0);
```



- **通道分割与合并**：`split()` 和 `merge()`。

```c++
cv::Mat channels[3];
cv::split(img, channels);  // 分割图像
cv::merge(channels, 3, img);  // 合并图像
```



**注意事项：**

- 在处理高分辨率视频时，`VideoCapture` 和 `VideoWriter` 的性能可能会成为瓶颈，建议考虑多线程或 GPU 加速。





### 3. 图像基本操作
- **颜色空间转换**：`cvtColor()`，RGB、BGR、HSV、灰度图等颜色空间之间的转换。
- **图像缩放和翻转**：`resize()`，`flip()`，图像的放大缩小、水平或垂直翻转。
- **图像裁剪与拼接**：矩阵的 ROI（Region of Interest）操作，如何对图像进行裁剪和拼接。
- **几何变换**：`warpAffine()`，`warpPerspective()`，仿射变换、透视变换。

### 4. 图像处理

#### 4.1 平滑和滤波

- **平滑与滤波**：`blur()`，`GaussianBlur()`，`medianBlur()`，`bilateralFilter()`，图像的平滑与去噪。

常见的滤波操作包括均值滤波、Gaussian滤波、中值滤波、双边滤波等等

```c++
cv::GaussianBlur(img, img, cv::Size(5,5 ), 0);
```

##### 4.1.1 概念

**平滑**和**滤波**是图像处理中的基本操作，用于减少图像中的噪声、细节和变化，使图像看起来更加平滑或柔和。这些操作在图像预处理、特征提取，边缘检测等应用中都非常重要。

- **噪声**：图像中的随机误差，通常由传感器噪声、环境干扰等因素造成。常见噪声类型有高斯噪声、椒盐噪声等等。
- **平滑**：减少图像中的高频成分，使图像中的整体外观更加平滑。

##### 4.1.2 作用

- **去噪**：再进行后续处理（如边缘检测、图像分割）之前，使用平滑操作去除噪声，提高处理效果。
- **改善视觉效果**：在图像中减少不必要的细节，突出主要特征。
- **防止伪影**：在某些算法（如轮廓检测）中，平滑可以帮助减少假轮廓的产生。



##### 4.1.3 常见的平滑与滤波方法

###### 1. 均值滤波

- **原理**：通过计算每个像素及其邻域像素的平均值来平滑图像，抑制高频噪声。

- **实现**：使用 `cv::blur()` 或 `cv::GaussianBlur()`。
- **代码示例**：

```c++
cv::Mat img; // 输入图像
cv::Mat smoothed;
cv::blur(img, smoothed, cv::Size(5, 5)); // 5x5 的均值滤波
```





###### 2. 高斯滤波

- **原理**：使用高斯函数对像素进行加权平均，能够更好地保留边缘信息。

- **实现**：使用 `cv::GaussianBlur()`。
- **代码示例**：

```c++
cv::GaussianBlur(img, smoothed, cv::Size(5, 5), 0); // 5x5 高斯滤波
```



###### 3. 中值滤波

- **原理**：通过取邻域内像素值的中值来平滑图像，特别有效于去除椒盐噪声。

- **实现**：使用 `cv::medianBlur()`。
- **代码示例**：

```c++
cv::medianBlur(img, smoothed, 5); // 5x5 中值滤波
```



###### 4. 双边滤波

- **原理**：同时考虑空间距离和像素值差异，能够有效去除噪声的同时保持边缘。

- **实现**：使用 `cv::bilateralFilter()`。
- **代码示例**：

```c++
cv::bilateralFilter(img, smoothed, 9, 75, 75); // 双边滤波
```



##### 4.1.4 注意事项

1. 选择合适的滤波器

- 对于高斯噪声，通常使用高斯滤波

- 对于椒盐噪声，中值滤波效果更好
- 双边滤波适合在去噪同时保留边缘的场景。



------

⭐`如何确定噪声类型？`

1. 视觉观察

- **观察图像特征**
	- **高斯噪声**：在整个图像中均匀分布，通常表现为亮度变化，噪声较为细腻，通常难以察觉。
	- **椒盐噪声**：在图像中会随机出现白色（盐）和黑色（椒）像素点，显著影响图像的视觉效果。
	- **斑点噪声**：常见于低光照图像，通常在图像中的特定区域（如阴影部分）比较明显，表现为不规则的亮点。



2. 使用统计分析

- 直方图分析
	- 通过查看图像的直方图，判断像素值分布的异常情况。
	- 高斯噪声的直方图通常是连续的，而椒盐噪声则表现为直方图中存在极端值的尖峰。
- 噪声模拟分析
	- 进行小区域（窗口）内的统计分析，计算均值和方差。高斯噪声通常表现为均值稳定而方差较小，而椒盐噪声则会使得像素值极端化。



3. 应用滤波技术

- 尝试不同的滤波技术
	- 可以通过不同类型的滤波器处理图像，观察去噪效果。
	- 如果高斯滤波有效，说明可能是高斯噪声；如果中值滤波表现更好，则可能是椒盐噪声。





4. 使用图像处理算法

- 频域分析
	- 对图像进行傅里叶变换，分析频率成分。
	- 高斯噪声在频域上会表现为较为平滑的分布，而椒盐噪声则会引起高频成分的突变。



5. 参考文献与已知数据

- 查阅相关文献，了解常见噪声类型的特征和样本图像，对照自己的图像进行分析。



**总结：**

确定噪声类型并不总是简单的，有时可能需要结合多种方法进行综合判断。通过视觉观察、统计分析、滤波处理和频域分析，逐步缩小噪声类型的范围，从而选择合适的去噪策略。



------





2. 滤波器大小的选择

- 滤波器大小的选择（如 `cv::Size(5, 5)`）对结果有直接影响，过小可能效果不明显，过大则可能导致图像模糊。
- 需要根据具体的图像和噪声情况进行调整。



3. 边界处理

- 不同滤波器在处理图像边界时可能会产生不同的结果，需要注意边界效果。
- OpenCV 的滤波函数提供了不同的边界扩展方法（如 `BORDER_CONSTANT`, `BORDER_REFLECT` 等）。



4. 性能考虑

- 对于大图像或复杂滤波操作，计算量较大，可以考虑使用GPU加速或优化算法。





##### 4.1.5 示例应用

假设我们有一幅图像，其中包含了椒盐噪声，我们可以使用中值滤波来去噪。



```c++
#include <opencv2/opencv.hpp>

int main() {
    // 读取图像
    cv::Mat img = cv::imread("noisy_image.jpg");
    cv::Mat denoised;

    // 使用中值滤波去噪
    cv::medianBlur(img, denoised, 5); // 5x5 中值滤波

    // 显示结果
    cv::imshow("Original Image", img);
    cv::imshow("Denoised Image", denoised);
    cv::waitKey(0);

    return 0;
}
```

在这个示例中，我们读取了一幅含噪声的图像，通过中值滤波去噪，并显示原图与去噪后的图像。这样的处理在实际应用中能显著提高后续图像处理的效果。



#### 4.2 边缘检测

- **边缘检测**：`Canny()` 边缘检测，`Sobel()`，`Laplacian()`。

变元检测常用于提取图像中的边缘：

```
cv::Canny(img, edge, 100, 200);
```



- **形态学操作**：`erode()`，`dilate()`，`morphologyEx()`，腐蚀、膨胀和其他形态学变换。

形态学操作包括腐蚀(erode)和膨胀(dilate)，用于消除噪声或增强结构。

```c++
cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
cv::erode(img, img, element);
```



- **直方图与直方图均衡化**：`calcHist()`，`equalizeHist()`，直方图计算与均衡化。

计算图像的直方图并进行均衡化：

```c++
cv::Mat hist;
cv::calcHist(&img, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
cv::equalizeHist(gray_img, equalized_img);
```



- **阈值处理**：`threshold()`，自适应阈值 `adaptiveThreshold()`，OTSU 二值化。

```c++
cv::threshold(gray_img, binary_img, 128, 255, cv::THRESH_BINARY);
```









### 5. 图像特征
- **轮廓检测**：`findContours()`，`drawContours()`，轮廓检测、绘制与特征提取。
- **霍夫变换**：`HoughLines()`，`HoughCircles()`，霍夫线变换与霍夫圆变换。
- **角点检测**：`goodFeaturesToTrack()`，Harris 角点检测。
- **SIFT/SURF/ORB**：特征点检测与匹配（SIFT, SURF, ORB）。

### 6. 图像分割与对象检测
- **GrabCut 算法**：前景与背景分割算法。
- **分水岭算法**：基于梯度的图像分割算法。
- **模板匹配**：`matchTemplate()`，模板匹配算法。
- **轮廓分析**：形状匹配，计算面积、周长、最小外接矩形、最小外接圆等。

### 7. 视频处理
- **帧处理**：如何从视频中逐帧处理，`VideoCapture`。
- **运动检测与跟踪**：背景减法 `BackgroundSubtractorMOG2`，光流法 `calcOpticalFlowPyrLK()`。
- **对象跟踪**：KCF、CSRT、BOOSTING 等跟踪算法。

### 8. 图像匹配与变换
- **特征匹配**：`BFMatcher`，`FlannBasedMatcher`，基于关键点的图像匹配。
- **单应性**：`findHomography()`，利用单应矩阵做图像配准与透视变换。
- **立体视觉与深度估计**：双目相机标定、立体匹配、深度图生成。
- **图像拼接**：多图像拼接与全景图生成。

### 9. 相机标定与 3D 重建
- **相机标定**：`calibrateCamera()`，`solvePnP()`，相机内外参数估计。
- **立体匹配与 3D 重建**：双目立体匹配生成视差图，`StereoBM`，`StereoSGBM`。
- **3D 点云与深度图**：利用视差图生成 3D 点云，深度图。

### 10. 深度学习与 DNN 模块
- **加载预训练模型**：`dnn::readNetFromTensorflow()`，`readNetFromCaffe()`，`readNetFromONNX()`。
- **推理与检测**：利用预训练深度学习模型进行目标检测、分类、语义分割。
- **YOLO、SSD、Faster R-CNN**：流行的深度学习目标检测模型的使用。

### 11. 高级话题
- **多线程与异步处理**：使用 C++ 多线程与 OpenCV 的 `VideoCapture`、`VideoWriter` 异步读取与保存视频。
- **GPU 加速**：使用 OpenCV 的 CUDA 模块加速计算，`cv::cuda::GpuMat`。
- **优化技巧**：如何通过减少拷贝、优化内存使用来提升性能。

### 12. 实战项目
- **实时人脸检测**：使用 Haar 级联分类器或 DNN 模型进行人脸检测。
- **车辆检测与计数**：基于视频流的车辆检测与交通监控应用。
- **姿态估计与 AR**：基于 SolvePnP 的物体姿态估计和增强现实应用。

### 13. 调试与性能优化
- **OpenCV 调试**：如何使用 OpenCV 的调试工具与日志系统。
- **性能评估与优化**：图像处理的性能分析，`getTickCount()`，优化内存访问与算法实现。

### 14. 常见问题解决
- **Mat 类常见问题**：图像内存分配、Mat 的深拷贝与浅拷贝。
- **异常处理**：常见异常的捕获与处理。

这些知识点构成了使用 C++ 结合 OpenCV 进行计算机视觉开发的全面基础与进阶内容，学习者可以根据项目需求逐步掌握各个模块。