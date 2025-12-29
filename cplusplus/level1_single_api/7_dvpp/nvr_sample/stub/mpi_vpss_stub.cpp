#include "hi_mpi_vpss.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
hi_s32 hi_mpi_vpss_create_grp(hi_vpss_grp grp, const hi_vpss_grp_attr *grp_attr)
{
    return 0;
}

hi_s32 hi_mpi_vpss_destroy_grp(hi_vpss_grp grp)
{
    return 0;
}

hi_s32 hi_mpi_vpss_start_grp(hi_vpss_grp grp)
{
    return 0;
}

hi_s32 hi_mpi_vpss_stop_grp(hi_vpss_grp grp)
{
    return 0;
}

hi_s32 hi_mpi_vpss_set_grp_crop(hi_vpss_grp grp, const hi_vpss_crop_info *crop_info)
{
    return 0;
}

hi_s32 hi_mpi_vpss_get_grp_crop(hi_vpss_grp grp, hi_vpss_crop_info *crop_info)
{
    return 0;
}

hi_s32 hi_mpi_vpss_set_grp_fisheye_cfg(hi_vpss_grp grp, const hi_fisheye_cfg *fisheye_cfg)
{
    return 0;
}

hi_s32 hi_mpi_vpss_get_grp_fisheye_cfg(hi_vpss_grp grp, hi_fisheye_cfg *fisheye_cfg)
{
    return 0;
}

hi_s32 hi_mpi_vpss_set_chn_attr(hi_vpss_grp grp, hi_vpss_chn chn, const hi_vpss_chn_attr *chn_attr)
{
    return 0;
}

hi_s32 hi_mpi_vpss_get_chn_attr(hi_vpss_grp grp, hi_vpss_chn chn, hi_vpss_chn_attr *chn_attr)
{
    return 0;
}

hi_s32 hi_mpi_vpss_enable_chn(hi_vpss_grp grp, hi_vpss_chn chn)
{
    return 0;
}

hi_s32 hi_mpi_vpss_disable_chn(hi_vpss_grp grp, hi_vpss_chn chn)
{
    return 0;
}

hi_s32 hi_mpi_vpss_get_chn_frame(hi_vpss_grp grp, hi_vpss_chn chn,
    hi_video_frame_info *video_frame, hi_s32 milli_sec)
{
    return 0;
}

hi_s32 hi_mpi_vpss_release_chn_frame(hi_vpss_grp grp, hi_vpss_chn chn,
    const hi_video_frame_info *video_frame)
{
    return 0;
}

hi_s32 hi_mpi_vpss_set_grp_param(hi_vpss_grp grp, const hi_vpss_grp_param *grp_param)
{
    return 0;
}

hi_s32 hi_mpi_vpss_get_grp_param(hi_vpss_grp grp, hi_vpss_grp_param *grp_param)
{
    return 0;
}

hi_s32 hi_mpi_vpss_set_chn_fisheye(hi_vpss_grp grp, hi_vpss_chn chn, const hi_fisheye_correction_attr *correction_attr)
{
    return 0;
}

hi_s32 hi_mpi_vpss_get_chn_fisheye(hi_vpss_grp grp, hi_vpss_chn chn, hi_fisheye_correction_attr *correction_attr)
{
    return 0;
}

hi_s32 hi_mpi_vpss_get_chn_fd(hi_vpss_grp grp, hi_vpss_chn chn)
{
    return 0;
}
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif