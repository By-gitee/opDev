/**
* @file main.cpp
*
* Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include "data_utils.h"
#include "acl/acl.h"
#include "add_custom_tiling.h"
#include "aclrtlaunch_add_custom.h"
extern AddCustomTiling* GenerateAddCustomTiling(uint32_t totalLength);
int32_t main(int32_t argc, char *argv[])
{
   uint32_t blockDim = 8;
   size_t inputByteSize = 8 * 2048 * sizeof(uint16_t);
   size_t outputByteSize = 8 * 2048 * sizeof(uint16_t);
   size_t totalLength = 8 * 2048;

   CHECK_ACL(aclInit(nullptr));
   int32_t deviceId = 0;
   CHECK_ACL(aclrtSetDevice(deviceId));
   aclrtStream stream = nullptr;
   CHECK_ACL(aclrtCreateStream(&stream));

   void *xHost, *yHost, *zHost;
   void *xDevice, *yDevice, *zDevice;

   CHECK_ACL(aclrtMallocHost((void **)(&xHost), inputByteSize));
   CHECK_ACL(aclrtMallocHost((void **)(&yHost), inputByteSize));
   CHECK_ACL(aclrtMallocHost((void **)(&zHost), outputByteSize));
   CHECK_ACL(aclrtMalloc((void **)&xDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
   CHECK_ACL(aclrtMalloc((void **)&yDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
   CHECK_ACL(aclrtMalloc((void **)&zDevice, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));

   ReadFile("./input/input_x.bin", inputByteSize, xHost, inputByteSize);
   ReadFile("./input/input_y.bin", inputByteSize, yHost, inputByteSize);

   CHECK_ACL(aclrtMemcpy(xDevice, inputByteSize, xHost, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
   CHECK_ACL(aclrtMemcpy(yDevice, inputByteSize, yHost, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
   
   AddCustomTiling* tiling = GenerateAddCustomTiling(totalLength);
   ACLRT_LAUNCH_KERNEL(add_custom)
   (blockDim, stream, xDevice, yDevice, zDevice,tiling);
   CHECK_ACL(aclrtSynchronizeStream(stream));

   CHECK_ACL(aclrtMemcpy(zHost, outputByteSize, zDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST));
   WriteFile("./output/output_z.bin", zHost, outputByteSize);

   CHECK_ACL(aclrtFree(xDevice));
   CHECK_ACL(aclrtFree(yDevice));
   CHECK_ACL(aclrtFree(zDevice));
   CHECK_ACL(aclrtFreeHost(xHost));
   CHECK_ACL(aclrtFreeHost(yHost));
   CHECK_ACL(aclrtFreeHost(zHost));

   CHECK_ACL(aclrtDestroyStream(stream));
   CHECK_ACL(aclrtResetDevice(deviceId));
   CHECK_ACL(aclFinalize());
   return 0;
}
