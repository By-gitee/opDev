#include "hi_mpi_vo.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
hi_s32 hi_mpi_vo_set_pub_attr(hi_vo_dev dev, const hi_vo_pub_attr *pub_attr)
{
    return 0;
}
hi_s32 hi_mpi_vo_enable(hi_vo_dev dev)
{
    return 0;
}
hi_s32 hi_mpi_vo_disable(hi_vo_dev dev)
{
    return 0;
}
hi_s32 hi_mpi_vo_set_video_layer_attr(hi_vo_layer layer, const hi_vo_video_layer_attr *layer_attr)
{
    return 0;
}
hi_s32 hi_mpi_vo_get_video_layer_attr(hi_vo_layer layer, hi_vo_video_layer_attr *layer_attr)
{
    return 0;
}
hi_s32 hi_mpi_vo_enable_video_layer(hi_vo_layer layer)
{
    return 0;
}
hi_s32 hi_mpi_vo_disable_video_layer(hi_vo_layer layer)
{
    return 0;
}
hi_s32 hi_mpi_vo_enable_chn(hi_vo_layer layer, hi_vo_chn chn)
{
    return 0;
}
hi_s32 hi_mpi_vo_disable_chn(hi_vo_layer layer, hi_vo_chn chn)
{
    return 0;
}
hi_s32 hi_mpi_vo_set_chn_attr(hi_vo_layer layer, hi_vo_chn chn, const hi_vo_chn_attr *chn_attr)
{
    return 0;
}
hi_s32 hi_mpi_vo_set_chn_param(hi_vo_layer layer, hi_vo_chn chn, const hi_vo_chn_param *chn_param)
{
    return 0;
}
hi_s32 hi_mpi_vo_send_frame(hi_vo_layer layer, hi_vo_chn chn, const hi_video_frame_info *frame_info,
    hi_s32 milli_sec)
{
    return 0;
}
hi_s32 hi_mpi_vo_pause_chn(int layer, int chn)
{
    return 0;
}
hi_s32 hi_mpi_vo_resume_chn(int layer, int chn)
{
    return 0;
}
hi_s32 hi_mpi_vo_create_pool(const hi_u64 size)
{
    return 0;
}
hi_u64 hi_mpi_vo_handle_to_phys_addr(hi_s32 handle)
{
    return 0;
}
hi_s32 hi_mpi_vo_destroy_pool(hi_s32 handle)
{
    return 0;
}
hi_s32 hi_mpi_vo_hide_chn(hi_vo_layer layer, hi_vo_chn chn)
{
    return 0;
}
hi_s32 hi_mpi_vo_show_chn(hi_vo_layer layer, hi_vo_chn chn)
{
    return 0;
}
hi_s32 hi_mpi_vo_set_chn_frame_rate(hi_vo_layer layer, hi_vo_layer chn, hi_s32 frame_rate)
{
    return 0;
}
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif