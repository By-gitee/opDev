#include "sample_comm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>

aclrtContext g_context = HI_NULL;

/* The order of g_sample_pic_size's element must be consistent with the enum value defined in "hi_pic_size". */
static hi_size g_sample_pic_size[PIC_BUTT] = {
    { 352,  288  },  /* PIC_CIF */
    { 640,  360  },  /* PIC_360P */
    { 720,  576  },  /* PIC_D1_PAL */
    { 720,  480  },  /* PIC_D1_NTSC */
    { 960,  576  },  /* PIC_960H */
    { 1280, 720  },  /* PIC_720P */
    { 1920, 1080 },  /* PIC_1080P */
    { 720,  480  },  /* PIC_480P */
    { 720,  576  },  /* PIC_576P */
    { 800,  600  },  /* PIC_800X600 */
    { 1024, 768  },  /* PIC_1024X768 */
    { 1280, 1024 },  /* PIC_1280X1024 */
    { 1366, 768  },  /* PIC_1366X768 */
    { 1440, 900  },  /* PIC_1440X900 */
    { 1280, 800  },  /* PIC_1280X800 */
    { 1600, 1200 },  /* PIC_1600X1200 */
    { 1680, 1050 },  /* PIC_1680X1050 */
    { 1920, 1200 },  /* PIC_1920X1200 */
    { 640,  480  },  /* PIC_640X480 */
    { 1920, 2160 },  /* PIC_1920X2160 */
    { 2560, 1440 },  /* PIC_2560X1440 */
    { 2560, 1600 },  /* PIC_2560X1600 */
    { 2592, 1520 },  /* PIC_2592X1520 */
    { 2592, 1944 },  /* PIC_2592X1944 */
    { 3840, 2160 },  /* PIC_3840X2160 */
    { 4096, 2160 },  /* PIC_4096X2160 */
    { 3000, 3000 },  /* PIC_3000X3000 */
    { 4000, 3000 },  /* PIC_4000X3000 */
    { 7680, 4320 },  /* PIC_7680X4320 */
    { 3840, 8640 }   /* PIC_3840X8640 */
};

hi_void sample_comm_sys_exit(hi_void)
{
    hi_mpi_sys_exit();
    return;
}

hi_s32 sample_comm_sys_get_pic_size(hi_pic_size pic_size, hi_size *size)
{
    if (size == HI_NULL) {
        sample_print("null ptr arg!\n");
        return HI_FAILURE;
    }

    if (pic_size >= PIC_BUTT) {
        sample_print("illegal pic_size!\n");
        return HI_FAILURE;
    }

    size->width = g_sample_pic_size[pic_size].width;
    size->height = g_sample_pic_size[pic_size].height;

    return HI_SUCCESS;
}

hi_s32 sample_comm_sys_init()
{
    int32_t s32Ret = HI_FAILURE;    
    aclError aclRet = aclInit(HI_NULL);
    if (aclRet != ACL_SUCCESS) {
        sample_print("[%s][%d] aclInit failed, error code = %d.\n", __FUNCTION__, __LINE__, aclRet);
        return HI_FAILURE;
    }
    sample_print("[%s][%d] aclInit Success.\n", __FUNCTION__, __LINE__);

    aclRet = aclrtSetDevice(0);
    if (aclRet != ACL_SUCCESS) {
        sample_print("[%s][%d] aclrtSetDevice 0 failed, error code = %d.\n", __FUNCTION__, __LINE__, aclRet);
        aclFinalize();
        return HI_FAILURE;
    }
    sample_print("[%s][%d] aclrtSetDevice 0 Success.\n", __FUNCTION__, __LINE__);

    aclRet = aclrtCreateContext(&g_context, 0);
    if (aclRet != ACL_SUCCESS) {
        sample_print("[%s][%d] aclrtCreateContext failed, error code = %d.\n", __FUNCTION__, __LINE__, aclRet);
        aclrtResetDevice(0);
        aclFinalize();
        return HI_FAILURE;
    }
    sample_print("[%s][%d] aclrtCreateContext Success\n", __FUNCTION__, __LINE__);

    s32Ret = hi_mpi_sys_init();
    if (HI_SUCCESS != s32Ret) {
        SAMPLE_PRT("hi_mpi_sys_init failed!\n");
        return HI_FAILURE;
    }
    SAMPLE_PRT("hi_mpi_sys_init succeed!\n");
    return HI_SUCCESS;
}

hi_s32 sample_comm_vpss_bind_vo(hi_vpss_grp vpss_grp, hi_vpss_chn vpss_chn, hi_vo_layer vo_layer, hi_vo_chn vo_chn)
{
    hi_mpp_chn src_chn;
    hi_mpp_chn dest_chn;

    src_chn.mod_id = HI_ID_VPSS;
    src_chn.dev_id = vpss_grp;
    src_chn.chn_id = vpss_chn;

    dest_chn.mod_id = HI_ID_VO;
    dest_chn.dev_id = vo_layer;
    dest_chn.chn_id = vo_chn;

    check_return(hi_mpi_sys_bind(&src_chn, &dest_chn), "hi_mpi_sys_bind(VPSS-VO)");

    return HI_SUCCESS;
}

hi_s32 sample_comm_vpss_un_bind_vo(hi_vpss_grp vpss_grp, hi_vpss_chn vpss_chn, hi_vo_layer vo_layer, hi_vo_chn vo_chn)
{
    hi_mpp_chn src_chn;
    hi_mpp_chn dest_chn;

    src_chn.mod_id = HI_ID_VPSS;
    src_chn.dev_id = vpss_grp;
    src_chn.chn_id = vpss_chn;

    dest_chn.mod_id = HI_ID_VO;
    dest_chn.dev_id = vo_layer;
    dest_chn.chn_id = vo_chn;

    check_return(hi_mpi_sys_unbind(&src_chn, &dest_chn), "hi_mpi_sys_unbind(VPSS-VO)");

    return HI_SUCCESS;
}