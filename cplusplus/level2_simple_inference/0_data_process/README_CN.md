中文|[English](README.md)

#  数据处理相关案例

#### 介绍
本目录包含Atlas支持的媒体数据处理功能，各文件夹对应不同功能，以供用户参考。目录结构与具体说明如下。
| 样例  | 说明  | 支持芯片 |
|---|---|---|
| [batchcrop](./batchcrop)  | batchcrop接口，抠图，一图多框  | Atlas 200/300/500 推理产品，Atlas 推理系列产品（配置Ascend 310P AI处理器），Atlas 训练系列产品 |
| [crop](./crop)  | crop接口，从输入图片中抠出需要用的图片区域  | Atlas 200/300/500 推理产品 |
| [cropandpaste](./cropandpaste)  | cropandpaste接口，从输入图片中抠出来的图，对抠出的图进行缩放后，放在用户输出图片的指定区域 | Atlas 200/300/500 推理产品 |
| [ffmpegdecode](./ffmpegdecode) | 调用ffmpeg接口实现视频切帧功能样例 | Atlas 200/300/500 推理产品 |
| [jpegd](./jpegd)  | jpegd接口，实现.jpg、.jpeg、.JPG、.JPEG图片的解码  | Atlas 200/300/500 推理产品 |
| [jpege](./jpege)  | jpege接口，将YUV格式图片编码成JPEG压缩格式的图片文件  | Atlas 200/300/500 推理产品 |
| [resize](./resize)  | resize接口。针对图像做缩放操作  | Atlas 200/300/500 推理产品 |
| [smallResolution_cropandpaste](./smallResolution_cropandpaste)  | cropandpaste接口。对指定输入图片进行抠图，再贴图到输出图片中  | Atlas 200/300/500 推理产品，Atlas 推理系列产品（配置Ascend 310P AI处理器），Atlas 训练系列产品 |
| [vdec](./vdec)  | vdec接口，实现视频的解码，输出YUV420SP格式（包括NV12和NV21）的图片  | Atlas 200/300/500 推理产品 |
| [vdecandvenc](./vdecandvenc)  | vdec接口和venc接口，调用dvpp的venc和vdec接口，实现视频编码功能  | Atlas 200/300/500 推理产品 |
| [venc](./venc) | venc接口，将原始mp4文件数据编码成H264/H265格式的视频码流 | Atlas 200/300/500 推理产品 |
| [venc_image](./venc_image) | venc接口，将一张YUV420SP NV12格式的图片连续编码n次，生成一个H265格式的视频码流 | Atlas 200/300/500 推理产品，Atlas 推理系列产品（配置Ascend 310P AI处理器） |