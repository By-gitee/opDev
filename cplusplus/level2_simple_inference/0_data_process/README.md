English|[中文](README_CN.md)

#  Data processing related sample

#### Directory structure and description
This catalog contains the media data processing functions supported by Atlas, and each folder corresponds to different functions for users' reference. The directory structure and specific instructions are as follows.
| Sample  | description  | support chip |
|---|---|---|
| [batchcrop](./batchcrop)  | Batch crop interface, cut out, one picture with multiple frames  | Atlas 200/300/500 inference product，Atlas inference products (with Ascend 310P AI Processors)，Atlas training products |
| [crop](./crop)  | Crop interface, cut out the image area that needs to be used from the input image  | Atlas 200/300/500 inference product |
| [cropandpaste](./cropandpaste)  | Cropandpaste interface, cut out the picture from the input picture, after the cut out picture is scaled, placed in the designated area of the user output picture | Atlas 200/300/500 inference product |
| [ffmpegdecode](./ffmpegdecode) | Example of calling ffmpeg interface to realize video frame cutting function | Atlas 200/300/500 inference product |
| [jpegd](./jpegd)  | jpegd interface, to achieve the decoding of .jpg, .jpeg, .JPG, .JPEG pictures  | Atlas 200/300/500 inference product |
| [jpege](./jpege)  | jpege interface, encoding YUV format pictures into picture files in JPEG compression format  | Atlas 200/300/500 inference product |
| [resize](./resize)  | resize interface. Zoom in and zoom out the image  | Atlas 200/300/500 inference product |
| [smallResolution_cropandpaste](./smallResolution_cropandpaste)  | cropandpaste interface. Cut out the specified input picture, and then paste the picture to the output picture  | Atlas 200/300/500 inference product，Atlas inference products (with Ascend 310P AI Processors)，Atlas training products |
| [vdec](./vdec)  | vdec interface, to achieve video decoding, output YUV420SP format (including NV12 and NV21) pictures  | Atlas 200/300/500 inference product |
| [vdecandvenc](./vdecandvenc)  | vdec interface and venc interface, call dvpp's venc and vdec interface to realize video encoding function  | Atlas 200/300/500 inference product |
| [venc](./venc) | venc interface, encode the original mp4 file data into a video stream in H264/H265 format | Atlas 200/300/500 inference product |
| [venc_image](./venc_image) | venc interface, encode a picture in YUV420SP NV12 format n times continuously to generate a video stream in H265 format | Atlas 200/300/500 inference product，Atlas inference products (with Ascend 310P AI Processors) |