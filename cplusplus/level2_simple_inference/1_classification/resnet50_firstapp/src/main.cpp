#include "acl/acl.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <map>
#include <math.h>

using namespace std;
int32_t deviceId = 0;
uint32_t modelId;
size_t pictureDataSize = 0;
void *pictureHostData;
void *pictureDeviceData;
aclmdlDataset *inputDataSet;
aclDataBuffer *inputDataBuffer;
aclmdlDataset *outputDataSet;
aclDataBuffer *outputDataBuffer;
aclmdlDesc *modelDesc;
size_t outputDataSize = 0;
void *outputDeviceData;
void *outputHostData;

void InitResource()
{
	aclError ret = aclInit(nullptr);
	ret = aclrtSetDevice(deviceId);
}

void LoadModel(const char* modelPath)
{
	aclError ret = aclmdlLoadFromFile(modelPath, &modelId);
}

//申请内存，使用C/C++标准库的函数将测试图片读入内存
void ReadPictureTotHost(const char *picturePath)
{
	string fileName = picturePath;
	ifstream binFile(fileName, ifstream::binary);
	binFile.seekg(0, binFile.end);
	pictureDataSize = binFile.tellg();
	binFile.seekg(0, binFile.beg);
	aclError ret = aclrtMallocHost(&pictureHostData, pictureDataSize);
	binFile.read((char*)pictureHostData, pictureDataSize);
	binFile.close();
}

//申请Device侧的内存，再以内存复制的方式将内存中的图片数据传输到Device
void CopyDataFromHostToDevice()
{
	aclError ret = aclrtMalloc(&pictureDeviceData, pictureDataSize, ACL_MEM_MALLOC_HUGE_FIRST);
	ret = aclrtMemcpy(pictureDeviceData, pictureDataSize, pictureHostData, pictureDataSize, ACL_MEMCPY_HOST_TO_DEVICE);
}

void LoadPicture(const char* picturePath)
{
	ReadPictureTotHost(picturePath);
	CopyDataFromHostToDevice();
}

// 准备模型推理的输入数据结构
void CreateModelInput()
{
    // 创建aclmdlDataset类型的数据，描述模型推理的输入
	inputDataSet = aclmdlCreateDataset();
	inputDataBuffer = aclCreateDataBuffer(pictureDeviceData, pictureDataSize);
	aclError ret = aclmdlAddDatasetBuffer(inputDataSet, inputDataBuffer);
}

// 准备模型推理的输出数据结构
void CreateModelOutput()
{
    // 创建模型描述信息
	modelDesc =  aclmdlCreateDesc();
	aclError ret = aclmdlGetDesc(modelDesc, modelId);
    // 创建aclmdlDataset类型的数据，描述模型推理的输出
	outputDataSet = aclmdlCreateDataset();
    // 获取模型输出数据需占用的内存大小，单位为Byte
	outputDataSize = aclmdlGetOutputSizeByIndex(modelDesc, 0);
    // 申请输出内存
	ret = aclrtMalloc(&outputDeviceData, outputDataSize, ACL_MEM_MALLOC_HUGE_FIRST);
	outputDataBuffer = aclCreateDataBuffer(outputDeviceData, outputDataSize);
	ret = aclmdlAddDatasetBuffer(outputDataSet, outputDataBuffer);
}

// 执行模型
void Inference()
{
    CreateModelInput();
	CreateModelOutput();
	aclError ret = aclmdlExecute(modelId, inputDataSet, outputDataSet);
}

void PrintResult()
{
        // 获取推理结果数据
        aclError ret = aclrtMallocHost(&outputHostData, outputDataSize);
        ret = aclrtMemcpy(outputHostData, outputDataSize, outputDeviceData, outputDataSize, ACL_MEMCPY_DEVICE_TO_HOST);
        // 将内存中的数据转换为float类型
        float* outFloatData = reinterpret_cast<float *>(outputHostData);

        // 屏显测试图片的top5置信度的类别编号
        map<float, unsigned int, greater<float>> resultMap;
        for (unsigned int j = 0; j < outputDataSize / sizeof(float);++j)
        {
                resultMap[*outFloatData] = j;
                outFloatData++;
        }

        // do data processing with softmax and print top 5 classes
        double totalValue=0.0;
        for (auto it = resultMap.begin(); it != resultMap.end(); ++it) {
            totalValue += exp(it->first);
        }

        int cnt = 0;
        for (auto it = resultMap.begin();it != resultMap.end();++it)
        {
                if(++cnt > 5)
                {
                        break;
                }
                printf("top %d: index[%d] value[%lf] \n", cnt, it->second, exp(it->first) /totalValue);
        }
}

void UnloadModel()
{
    // 释放模型描述信息
	aclmdlDestroyDesc(modelDesc);
    // 卸载模型
	aclmdlUnload(modelId);
}

void UnloadPicture()
{
	aclError ret = aclrtFreeHost(pictureHostData);
	pictureHostData = nullptr;
	ret = aclrtFree(pictureDeviceData);
	pictureDeviceData = nullptr;
	aclDestroyDataBuffer(inputDataBuffer);
	inputDataBuffer = nullptr;
	aclmdlDestroyDataset(inputDataSet);
	inputDataSet = nullptr;
	
	ret = aclrtFreeHost(outputHostData);
	outputHostData = nullptr;
	ret = aclrtFree(outputDeviceData);
	outputDeviceData = nullptr;
	aclDestroyDataBuffer(outputDataBuffer);
	outputDataBuffer = nullptr;
	aclmdlDestroyDataset(outputDataSet);
	outputDataSet = nullptr;
}

void DestroyResource()
{
	aclError ret = aclrtResetDevice(deviceId);
	aclFinalize();
}

int main()
{	
    // 1.定义一个资源初始化的函数，用于AscendCL初始化、运行管理资源申请（指定计算设备）
	InitResource();
	
    // 2.定义一个模型加载的函数，加载图片分类的模型，后续推理使用，若om文件不在model目录下，请根据实际存放目录修改此处的代码
	const char *modelPath = "../model/resnet50.om";
	LoadModel(modelPath);
	
    // 3.定义一个读图片数据的函数，将测试图片数据读入内存，并传输到Device侧，后续推理使用
    const char *picturePath = "../data/dog1_1024_683.bin";
	LoadPicture(picturePath);
	
    // 4.定义一个推理的函数，用于执行推理
	Inference();
	
    // 5.定义一个推理结果数据处理的函数，用于在终端上屏显测试图片的top5置信度的类别编号
	PrintResult();
	
    // 6.定义一个模型卸载的函数，卸载图片分类的模型
	UnloadModel();
	
    // 7.定义一个函数，用于释放内存、销毁推理相关的数据类型，防止内存泄露
	UnloadPicture();
	
    // 8.定义一个资源去初始化的函数，用于AscendCL去初始化、运行管理资源释放（释放计算设备）
	DestroyResource();
}
