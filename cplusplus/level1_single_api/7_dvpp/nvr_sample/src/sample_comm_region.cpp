#include "sample_comm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include "region_param_parser_ext.h"

#define OVERLAYEX_MIN_HANDLE 20
#define COVER_MIN_HANDLE 40

#define MOSAIC_MIN_HANDLE 100

static hi_s32 sample_comm_region_get_min_handle(hi_rgn_type type)
{
    hi_s32 min_handle;
    switch (type) {
        case HI_RGN_OVERLAYEX:
            min_handle = OVERLAYEX_MIN_HANDLE;
            break;
        case HI_RGN_COVER:
            min_handle = COVER_MIN_HANDLE;
            break;
        case HI_RGN_MOSAIC:
            min_handle = MOSAIC_MIN_HANDLE;
            break;
        default:
            min_handle = -1;
            break;
    }
    return min_handle;
}

static hi_s32 sample_region_create_mosaic(hi_s32 handle_num)
{
    hi_s32 ret;
    hi_s32 i;
    hi_rgn_attr region;

    region.type = HI_RGN_MOSAIC;

    for (i = MOSAIC_MIN_HANDLE; i < MOSAIC_MIN_HANDLE + handle_num; i++) {
        ret = hi_mpi_rgn_create(i, &region);
        if (ret != HI_SUCCESS) {
            sample_print("hi_mpi_rgn_create failed with %#x!\n", ret);
            return HI_FAILURE;
        }
    }

    return HI_SUCCESS;
}

hi_s32 sample_comm_region_create(hi_s32 handle_num, hi_rgn_type type)
{
    hi_s32 ret = HI_SUCCESS;
    if (handle_num <= 0 || handle_num > 16) { /* 16:max_num */
        sample_print("handle_num is illegal %d!\n", handle_num);
        return HI_FAILURE;
    }
    switch (type) {
        case HI_RGN_MOSAIC:
            ret = sample_region_create_mosaic(handle_num);
            break;
        default:
            break;
    }
    if (ret != HI_SUCCESS) {
        sample_print("sample_comm_region_create failed! handle_num%d,type:%d!\n", handle_num, type);
        return HI_FAILURE;
    }
    return ret;
}

static hi_void sample_region_get_mosaic_chn_attr(hi_s32 handle, hi_rgn_mosaic_chn_attr *mosaic_chn)
{
    hi_s32 ret;
    hi_u32 index = handle - MOSAIC_MIN_HANDLE;

    (void)get_mosaic_attr_by_index(index, mosaic_chn);
}

static hi_s32 sample_region_attach_to_chn(hi_rgn_handle handle, hi_mpp_chn *chn, hi_rgn_chn_attr *chn_attr)
{
    hi_s32 ret;

    ret = hi_mpi_rgn_attach_to_chn(handle, chn, chn_attr);
    if (ret != HI_SUCCESS) {
        sample_print("hi_mpi_rgn_attach_to_chn failed with %#x!\n", ret);
        return HI_FAILURE;
    }
    return HI_SUCCESS;
}

static hi_s32 sample_region_detach_from_chn(hi_rgn_handle handle, hi_mpp_chn *chn)
{
    hi_s32 ret;
    ret = hi_mpi_rgn_detach_from_chn(handle, chn);
    if (ret != HI_SUCCESS) {
        sample_print("hi_mpi_rgn_detach_from_chn failed with %#x!\n", ret);
        return HI_FAILURE;
    }
    return HI_SUCCESS;
}

hi_s32 sample_comm_region_detach_frm_chn(hi_s32 handle_num, hi_rgn_type type, hi_mpp_chn *mpp_chn)
{
    hi_s32 i;
    hi_s32 ret;
    hi_s32 min_handle;

    if (handle_num <= 0 || handle_num > 16) { /* 16:max region num */
        sample_print("handle_num is illegal %d!\n", handle_num);
        return HI_FAILURE;
    }
    if (mpp_chn == HI_NULL) {
        sample_print("mpp_chn is NULL !\n");
        return HI_FAILURE;
    }
    min_handle = sample_comm_region_get_min_handle(type);
    for (i = min_handle; i < min_handle + handle_num; i++) {
        ret = sample_region_detach_from_chn(i, mpp_chn);
        if (ret != HI_SUCCESS) {
            sample_print("sample_region_detach_from_chn failed! handle:%d\n", i);
        }
    }
    return HI_SUCCESS;
}

static hi_s32 sample_region_set_mosaic_chn_attr(hi_s32 handle_num, hi_rgn_chn_attr *chn_attr, hi_mpp_chn *mpp_chn)
{
    hi_s32 i;
    hi_s32 ret;
    chn_attr->type = HI_RGN_MOSAIC;
    for (i = MOSAIC_MIN_HANDLE; i < MOSAIC_MIN_HANDLE + handle_num; i++) {
        sample_region_get_mosaic_chn_attr(i, &chn_attr->attr.mosaic_chn);
        ret = sample_region_attach_to_chn(i, mpp_chn, chn_attr);
        if (ret != HI_SUCCESS) {
            sample_print("sample_region_attach_to_chn failed!\n");
            sample_comm_region_detach_frm_chn(i - MOSAIC_MIN_HANDLE + 1, HI_RGN_MOSAIC, mpp_chn);
            return ret;
        }
    }
    return HI_SUCCESS;
}

static hi_s32 sample_region_set_chn_attr(hi_s32 handle_num, hi_rgn_type type,
                                         hi_rgn_chn_attr *chn_attr, hi_mpp_chn *chn)
{
    hi_s32 ret = HI_ERR_RGN_ILLEGAL_PARAM;
    switch (type) {
        case HI_RGN_MOSAIC:
            ret = sample_region_set_mosaic_chn_attr(handle_num, chn_attr, chn);
            break;
        default:
            break;
    }
    return ret;
}

hi_s32 sample_comm_region_attach_to_chn(hi_s32 handle_num, hi_rgn_type type, hi_mpp_chn *mpp_chn)
{
    hi_s32 ret;
    hi_rgn_chn_attr chn_attr;

    (hi_void)memset(&chn_attr, 0, sizeof(hi_rgn_chn_attr));
    /* set the chn config */
    chn_attr.is_show = HI_TRUE;
    ret = sample_region_set_chn_attr(handle_num, type, &chn_attr, mpp_chn);
    if (ret != HI_SUCCESS) {
        sample_print("sample_region_attach_to_chn failed!\n");
    }
    return ret;
}