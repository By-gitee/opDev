#include "sample_comm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>

hi_s32 sample_common_vpss_start(hi_vpss_grp grp, const hi_bool *chn_enable, const hi_vpss_grp_attr *grp_attr,
                                const hi_vpss_crop_info *crop_info, const hi_vpss_chn_attr *chn_attr)
{
    hi_vpss_chn vpss_chn;
    hi_s32 ret;
    hi_s32 i;

    ret = hi_mpi_vpss_create_grp(grp, grp_attr);
    if (ret != HI_SUCCESS) {
        sample_print("hi_mpi_vpss_create_grp(grp:%d) failed with %#x!\n", grp, ret);
        return HI_FAILURE;
    }

    ret = hi_mpi_vpss_set_grp_crop(grp, crop_info);
    if (ret != HI_SUCCESS) {
        sample_print("hi_mpi_vpss_set_grp_crop fail for %#x!\n", ret);
        return ret;
    }

    ret = hi_mpi_vpss_start_grp(grp);
    if (ret != HI_SUCCESS) {
        sample_print("hi_mpi_vpss_start_grp failed with %#x\n", ret);
        return HI_FAILURE;
    }

    for (i = 0; i < VPSS_MAX_PHYS_CHN_NUM; ++i) {
        if (chn_enable[i] == HI_TRUE) {
            vpss_chn = i;
            ret = hi_mpi_vpss_set_chn_attr(grp, vpss_chn, &chn_attr[vpss_chn]);
            if (ret != HI_SUCCESS) {
                sample_print("hi_mpi_vpss_set_chn_attr failed with %#x\n", ret);
                return HI_FAILURE;
            }

            ret = hi_mpi_vpss_enable_chn(grp, vpss_chn);
            if (ret != HI_SUCCESS) {
                sample_print("hi_mpi_vpss_enable_chn failed with %#x\n", ret);
                return HI_FAILURE;
            }
        }
    }
    return HI_SUCCESS;
}

hi_s32 sample_common_vpss_stop(hi_vpss_grp grp, const hi_bool *chn_enable)
{
    hi_s32 i;
    hi_s32 ret;
    hi_vpss_chn vpss_chn;

    for (i = 0; i < VPSS_MAX_PHYS_CHN_NUM; ++i) {
        if (chn_enable[i] == HI_TRUE) {
            vpss_chn = i;
            ret = hi_mpi_vpss_disable_chn(grp, vpss_chn);
            if (ret != HI_SUCCESS) {
                sample_print("grp %u, vpss_chn %u failed with %#x!\n", grp, vpss_chn, ret);
                return HI_FAILURE;
            }
        }
    }

    ret = hi_mpi_vpss_stop_grp(grp);
    if (ret != HI_SUCCESS) {
        sample_print("failed with %#x!\n", ret);
        return HI_FAILURE;
    }

    ret = hi_mpi_vpss_destroy_grp(grp);
    if (ret != HI_SUCCESS) {
        sample_print("failed with %#x!\n", ret);
        return HI_FAILURE;
    }

    return HI_SUCCESS;
}