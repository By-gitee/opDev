/**
 *  Copyright [2021] Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <signal.h>
#include <getopt.h>
#include <exception>
#include <iostream>
#include <vector>
#include <thread>
#include "common.h"

using namespace std;
aclrtRunMode g_run_mode;

void sample_vpc_handle_sig(int32_t signo)
{
    if (SIGINT == signo || SIGTSTP == signo || SIGTERM == signo) {
        hi_mpi_sys_exit();
        SAMPLE_PRT("\033[0;31mprogram exit abnormally!\033[0;39m\n");
    }

    exit(0);
}

int32_t sample_comm_vpc_resize(FuncInput funcInput)
{
    if (g_run_mode == ACL_HOST) {
        aclError aclRet = aclrtSetCurrentContext(g_context);
        if (aclRet != ACL_SUCCESS) {
            SAMPLE_PRT("set current context failed, ret = %d\n", aclRet);
            return HI_FAILURE;
        }
    }

    // Get parameters
    char inputFileName[FILE_NAME_LEN];
    strcpy(inputFileName, funcInput.g_vpc_attribute.inputFileName);
    char outputFileName[FILE_NAME_LEN];
    strcpy(outputFileName, funcInput.g_vpc_attribute.outputFileName);
    uint32_t width = funcInput.g_vpc_attribute.width;
    uint32_t height = funcInput.g_vpc_attribute.height;
    uint32_t format = funcInput.g_vpc_attribute.format;
    uint32_t outWidth = funcInput.g_vpc_attribute.outWidth;
    uint32_t outHeight = funcInput.g_vpc_attribute.outHeight;
    uint32_t outFormat = funcInput.g_vpc_attribute.outFormat;
    double fx = funcInput.g_vpc_attribute.fx;
    double fy = funcInput.g_vpc_attribute.fy;
    int32_t interpolation = funcInput.g_vpc_attribute.interpolation;
    hi_vpc_chn chnId = funcInput.chnId;

    // Construct resize input image structure
    hi_vpc_pic_info inputPic;
    inputPic.picture_width = width;
    inputPic.picture_height = height;
    inputPic.picture_format = static_cast<hi_pixel_format>(format);

    configure_stride_and_buffer_size(inputPic);

    //// Construct resize output image structure
    hi_vpc_pic_info outputPic;
    outputPic.picture_width = outWidth;
    outputPic.picture_height = outHeight;
    if (outWidth == 0 || outHeight == 0) {
        outputPic.picture_width = static_cast<uint32_t>(width * fx + 0.5); // 0.5 is for rounding
        outputPic.picture_height = static_cast<uint32_t>(height * fy + 0.5); // 0.5 is for rounding
    }
    outputPic.picture_format = static_cast<hi_pixel_format>(outFormat);
    uint32_t dstBufferSize = configure_stride_and_buffer_size(outputPic);

    // Prepare to input data
    int32_t ret = prepare_input_data(inputPic, inputFileName);
    if (ret != HI_SUCCESS) {
        SAMPLE_PRT("prepare input data failed!\n");
        return ret;
    }

    // Create output memory
    ret = dvpp_mem_malloc(&outputPic.picture_address, outputPic.picture_buffer_size);
    if (ret != HI_SUCCESS) {
        SAMPLE_PRT("output buffer alloc failed!\n");
        dvpp_mem_free(inputPic.picture_address);
        inputPic.picture_address = nullptr;
        return HI_FAILURE;
    }
    memset_buffer(outputPic);

    //call the hi_mpi_vpc_resize interface for resizing processing
    uint32_t taskID = 0;
    ret = hi_mpi_vpc_resize(chnId, &inputPic, &outputPic, fx, fy, interpolation, &taskID, -1);
    if (ret != HI_SUCCESS) {
        SAMPLE_PRT("hi_mpi_vpc_resize failed, ret = %#x!\n", ret);
        dvpp_mem_free(inputPic.picture_address);
        inputPic.picture_address = nullptr;
        dvpp_mem_free(outputPic.picture_address);
        outputPic.picture_address = nullptr;
        return HI_FAILURE;
    }

    // Waiting for resizing processing to complete
    uint32_t taskIDResult = taskID;
    ret = hi_mpi_vpc_get_process_result(chnId, taskIDResult, -1);
    if (ret != HI_SUCCESS) {
        SAMPLE_PRT("hi_mpi_vpc_get_process_result failed, ret = %#x!\n", ret);
        dvpp_mem_free(inputPic.picture_address);
        inputPic.picture_address = nullptr;
        dvpp_mem_free(outputPic.picture_address);
        outputPic.picture_address = nullptr;
        return HI_FAILURE;
    }

    // Process the resized image and write it as a file
    ret = handle_output_data(outputPic, dstBufferSize, outputFileName);
    if (ret != HI_SUCCESS) {
        SAMPLE_PRT("handle_output_data failed!\n");
    }

    // release memory 
    dvpp_mem_free(inputPic.picture_address);
    inputPic.picture_address = nullptr;
    dvpp_mem_free(outputPic.picture_address);
    outputPic.picture_address = nullptr;
    return ret;
}

int main(int argc, char *argv[])
{
    // Capture abnormal exit events
    SAMPLE_PRT("signal input!\n");
    signal(SIGINT, sample_vpc_handle_sig);
    signal(SIGTERM, sample_vpc_handle_sig);

    //Set the relevant parameters. 
    // The default scenario for this example is to input a 1920 * 1080 YUV image and resize it to a size of 960 * 540。
    VpcAttr g_vpc_attribute;
    g_vpc_attribute.width = 1920;
    g_vpc_attribute.height = 1080;
    g_vpc_attribute.format = 1;
    g_vpc_attribute.outWidth = 960;
    g_vpc_attribute.outHeight = 540;
    g_vpc_attribute.outFormat = 1;
    g_vpc_attribute.fx = 0;
    g_vpc_attribute.fy = 0;
    g_vpc_attribute.interpolation = 0;
    strcpy(g_vpc_attribute.inputFileName, "../data/dvpp_vpc_1920x1080_nv12.yuv");
    strcpy(g_vpc_attribute.outputFileName, "resize_out.yuv");

    // Resource initialization
    int32_t s32Ret = get_run_mode();
    if (s32Ret != HI_SUCCESS) {
        return s32Ret;
    }
    else if (g_run_mode == ACL_HOST) {
        s32Ret = acl_init();
        if (s32Ret != HI_SUCCESS) {
            return s32Ret;
        }
    }
    s32Ret = hi_mpi_sys_init();
    if (s32Ret != HI_SUCCESS) {
        SAMPLE_PRT("hi_mpi_sys_init failed, ret = %#x!\n", s32Ret);
        if (g_run_mode == ACL_HOST) {
            acl_destory_resource();
        }
        return s32Ret;
    }

    //create vpc channel
    hi_vpc_chn chnId;
    hi_vpc_chn_attr stChnAttr {};
    stChnAttr.attr = 0;
    s32Ret = hi_mpi_vpc_sys_create_chn(&chnId, &stChnAttr);
    if (s32Ret != HI_SUCCESS) {
        SAMPLE_PRT("Call hi_mpi_vpc_sys_create_chn failed, ret = %#x\n", s32Ret);
        hi_mpi_sys_exit();
        if (g_run_mode == ACL_HOST) {
            acl_destory_resource();
        }
        return s32Ret;
    }
    
    //resize process
    FuncInput funcInput;
    funcInput.chnId = chnId;
    funcInput.g_vpc_attribute = g_vpc_attribute;
    sample_comm_vpc_resize(funcInput);

    // destroy channel.
    s32Ret = hi_mpi_vpc_destroy_chn(chnId);
    if (s32Ret != HI_SUCCESS) {
        SAMPLE_PRT("Call hi_mpi_vpc_destroy_chn failed, ret = %#x\n", s32Ret);
    }

    hi_mpi_sys_exit();
    if (g_run_mode == ACL_HOST) {
        acl_destory_resource();
    }

    SAMPLE_PRT("program exit\n");
    return s32Ret;
}
