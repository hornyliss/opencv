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
- **平滑与滤波**：`blur()`，`GaussianBlur()`，`medianBlur()`，`bilateralFilter()`，图像的平滑与去噪。
- **边缘检测**：`Canny()` 边缘检测，`Sobel()`，`Laplacian()`。
- **形态学操作**：`erode()`，`dilate()`，`morphologyEx()`，腐蚀、膨胀和其他形态学变换。
- **直方图与直方图均衡化**：`calcHist()`，`equalizeHist()`，直方图计算与均衡化。
- **阈值处理**：`threshold()`，自适应阈值 `adaptiveThreshold()`，OTSU 二值化。

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