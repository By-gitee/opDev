#ifndef __SAMPLE_COMM_H__
#define __SAMPLE_COMM_H__

#include <pthread.h>
#include <time.h>

#include "acl.h"

#include "hi_mpi_sys.h"
#include "hi_dvpp.h"
#include "hi_mpi_vpss.h"
#include "hi_mpi_vo.h"
#include "hi_mpi_hdmi.h"
#include "hi_region.h"

extern aclrtContext g_context;

#define  VPSS_MAX_PHYS_CHN_NUM      2

#define VPSS_AI_START_GRP 201
#define DISPLAY_NUM 1

#define HI_VO_LAYER_V0 0
#define HI_VO_LAYER_V1 1
#define HI_VO_LAYER_V2 2
#define HI_VO_LAYER_G0 4
#define HI_VO_LAYER_G1 5
#define HI_VO_LAYER_G3 7

#define SAMPLE_PRT(fmt...)   \
    do {\
        printf("[%s]-%d: ", __FUNCTION__, __LINE__);\
        printf(fmt);\
    }while(0)

#define  HI_VDEC_MAX_CHN_NUM  128
#define  HI_VPSS_MAX_GET_FRAME_CHN_NUM  260

#ifdef __cplusplus
extern "C" {
#endif /* end of #ifdef __cplusplus */

/* macro define */
#define FILE_NAME_LEN 128
#define FILE_PATH_LEN 128

#define SAMPLE_PIXEL_FORMAT HI_PIXEL_FORMAT_YVU_SEMIPLANAR_420

#define COLOR_RGB_RED      0xFF0000
#define COLOR_RGB_GREEN    0x00FF00
#define COLOR_RGB_BLUE     0x0000FF
#define COLOR_RGB_BLACK    0x000000
#define COLOR_RGB_YELLOW   0xFFFF00
#define COLOR_RGB_CYN      0x00ffff
#define COLOR_RGB_WHITE    0xffffff

#define SAMPLE_VO_DEV_DHD0 0                  /* VO's device HD0 */
#define SAMPLE_VO_DEV_DHD1 1                  /* VO's device HD1 */
#define SAMPLE_VO_DEV_DSD0 2                  /* VO's device SD0 */
#define SAMPLE_VO_DEV_UHD  SAMPLE_VO_DEV_DHD0 /* VO's ultra HD device:HD0 */
#define SAMPLE_VO_DEV_HD   SAMPLE_VO_DEV_DHD1 /* VO's HD device:HD1 */
#define SAMPLE_VO_LAYER_VHD0 0
#define SAMPLE_VO_LAYER_VHD1 1
#define SAMPLE_VO_LAYER_VHD2 2
#define SAMPLE_VO_LAYER_PIP  SAMPLE_VO_LAYER_VHD2
#define SAMPLE_RGN_HANDLE_NUM_MAX 16
#define SAMPLE_RGN_HANDLE_NUM_MIN 1


#define HI_ALIGN_UP(x, a)           ((((x) + ((a) - 1)) / (a)) * (a))
#define HI_ALIGN_DOWN(x, a)         (((x) / (a)) * (a))


#define sample_pause() \
    do { \
        printf("---------------press enter key to exit!---------------\n"); \
        getchar(); \
    } while (0)

#define sample_print(fmt...) \
    do { \
        printf("[%s]-%d: ", __FUNCTION__, __LINE__); \
        printf(fmt); \
    } while (0)

#define check_null_ptr_return(ptr) \
    do { \
        if ((ptr) == HI_NULL) { \
            printf("func:%s,line:%d, NULL pointer\n", __FUNCTION__, __LINE__); \
            return HI_FAILURE; \
        } \
    } while (0)

#define check_chn_return(express, chn, name) \
    do { \
        hi_s32 ret_ = (express); \
        if (ret_ != HI_SUCCESS) { \
            printf("\033[0;31m%s chn %d failed at %s: LINE: %d with %#x!\033[0;39m\n", \
                   (name), (chn), __FUNCTION__, __LINE__, ret_); \
            fflush(stdout); \
            return ret_; \
        } \
    } while (0)

#define check_return(express, name) \
    do { \
        hi_s32 ret_ = (express); \
        if (ret_ != HI_SUCCESS) { \
            printf("\033[0;31m%s failed at %s: LINE: %d with %#x!\033[0;39m\n", \
                   (name), __FUNCTION__, __LINE__, ret_); \
            return ret_; \
        } \
    } while (0)

typedef enum {
    PIC_CIF,
    PIC_360P,    /* 640 * 360 */
    PIC_D1_PAL,  /* 720 * 576 */
    PIC_D1_NTSC, /* 720 * 480 */
    PIC_960H,      /* 960 * 576 */
    PIC_720P,    /* 1280 * 720 */
    PIC_1080P,   /* 1920 * 1080 */
    PIC_480P,
    PIC_576P,
    PIC_800X600,
    PIC_1024X768,
    PIC_1280X1024,
    PIC_1366X768,
    PIC_1440X900,
    PIC_1280X800,
    PIC_1600X1200,
    PIC_1680X1050,
    PIC_1920X1200,
    PIC_640X480,
    PIC_1920X2160,
    PIC_2560X1440,
    PIC_2560X1600,
    PIC_2592X1520,
    PIC_2592X1944,
    PIC_3840X2160,
    PIC_4096X2160,
    PIC_3000X3000,
    PIC_4000X3000,
    PIC_6080X2800,
    PIC_7680X4320,
    PIC_3840X8640,
    PIC_BUTT
} hi_pic_size;

typedef enum {
    VO_MODE_1MUX = 0,
    VO_MODE_2MUX,
    VO_MODE_4MUX,
    VO_MODE_8MUX,
    VO_MODE_9MUX,
    VO_MODE_16MUX,
    VO_MODE_25MUX,
    VO_MODE_36MUX,
    VO_MODE_49MUX,
    VO_MODE_64MUX,
    VO_MODE_2X4,
    VO_MODE_BUTT
} sample_vo_mode;

typedef struct {
    hi_vo_intf_sync intf_sync;
    hi_u32 width;
    hi_u32 height;
    hi_u32 frame_rate;
} sample_vo_sync_info;

typedef struct {
    sample_vo_mode mode;
    hi_u32 wnd_num;
    hi_u32 square;
    hi_u32 row;
    hi_u32 col;
} sample_vo_wnd_info;

typedef struct {
    hi_u32 grp_num;
    sample_vo_mode mode;
} sample_vo_grp_vo_mode_info;


typedef struct {
    hi_hdmi_id hdmi_id;
    hi_hdmi_id hdmi1_id;
    /* for device */
    hi_vo_dev vo_dev;
    hi_vo_dev vo_layer;
    hi_vo_intf_type vo_intf_type;
    hi_vo_intf_sync intf_sync;
    hi_pic_size pic_size;
    hi_u32 bg_color;

    /* for layer */
    hi_pixel_format pix_format;
    hi_rect disp_rect;
    hi_size image_size;
    hi_vo_partition_mode vo_part_mode;
    hi_compress_mode compress_mode;

    hi_u32 dis_buf_len;
    hi_dynamic_range dst_dynamic_range;

    /* for channel */
    sample_vo_mode vo_mode;

    /* for user sync */
    hi_vo_sync_info sync_info;
    hi_vo_user_sync_info user_sync;
    hi_u32 dev_frame_rate;
    hi_vo_chn_param chn_param;
} sample_vo_cfg;

typedef struct {
    hi_pic_size pic_size;
    hi_vo_intf_sync intf_sync;
    hi_vo_intf_type intf_type;
} vdec_display_cfg;

typedef enum {
    THREAD_CTRL_START,
    THREAD_CTRL_PAUSE,
    THREAD_CTRL_STOP,
} thread_contrl;

typedef struct {
    hi_s32 chn_id;
    hi_payload_type type;
    hi_char c_file_path[FILE_PATH_LEN];
    hi_char c_file_name[FILE_NAME_LEN];
    hi_s32 stream_mode;
    hi_s32 milli_sec;
    hi_s32 min_buf_size;
    hi_s32 interval_time;
    thread_contrl e_thread_ctrl;
    hi_u64 pts_init;
    hi_u64 pts_increase;
    hi_bool circle_send;
    hi_u64 last_time;
    hi_u64 time_gap;
    hi_u64 fps;
} vdec_thread_param;

typedef struct {
    hi_video_dec_mode dec_mode;
    hi_u32 ref_frame_num;
    hi_data_bit_width bit_width;
} sample_vdec_video_attr;

typedef struct {
    hi_pixel_format pixel_format;
    hi_u32 alpha;
} sample_vdec_pic_attr;

typedef struct {
    hi_payload_type type;
    hi_vdec_send_mode mode;
    hi_u32 width;
    hi_u32 height;
    hi_u32 frame_buf_cnt;
    hi_u32 display_frame_num;
    union {
        sample_vdec_video_attr sample_vdec_video; /* structure with video (h265/h264) */
        sample_vdec_pic_attr sample_vdec_picture; /* structure with picture (jpeg/mjpeg) */
    };
} sample_vdec_attr;


hi_s32 sample_comm_sys_get_pic_size(hi_pic_size pic_size, hi_size *size);
hi_s32 sample_comm_sys_init(hi_void);
hi_void sample_comm_sys_exit(hi_void);
hi_s32 sample_comm_region_create(hi_s32 handle_num, hi_rgn_type type);
hi_s32 sample_comm_region_attach_to_chn(hi_s32 handle_num, hi_rgn_type type, hi_mpp_chn *mpp_chn);
hi_s32 sample_comm_region_detach_frm_chn(hi_s32 handle_num, hi_rgn_type type, hi_mpp_chn *mpp_chn);
hi_s32 sample_common_vpss_start(hi_vpss_grp grp, const hi_bool *chn_enable, const hi_vpss_grp_attr *grp_attr,
    const hi_vpss_crop_info *crop_info, const hi_vpss_chn_attr *chn_attr);
hi_s32 sample_common_vpss_stop(hi_vpss_grp grp, const hi_bool *chn_enable);
hi_s32 sample_comm_vo_start_vo(sample_vo_cfg *vo_config);
hi_s32 sample_comm_vo_stop_vo(const sample_vo_cfg *vo_config);
hi_void sample_comm_vdec_set_send_usleep(hi_u32 send_usleep);
hi_u32 sample_comm_vdec_get_recv_frames_cnt(hi_u32 vdec_chn);
hi_s32 sample_comm_vdec_bind_vpss(hi_vdec_chn vdec_chn, hi_vpss_grp vpss_grp);
hi_s32 sample_comm_vdec_un_bind_vpss(hi_vdec_chn vdec_chn, hi_vpss_grp vpss_grp);
hi_s32 sample_comm_vpss_bind_vo(hi_vpss_grp vpss_grp, hi_vpss_chn vpss_chn, hi_vo_layer vo_layer, hi_vo_chn vo_chn);
hi_s32 sample_comm_vpss_un_bind_vo(hi_vpss_grp vpss_grp, hi_vpss_chn vpss_chn, hi_vo_layer vo_layer, hi_vo_chn vo_chn);
hi_void sample_comm_vdec_cmd_not_circle_send(hi_u32 chn_num, vdec_thread_param *vdec_send,
    pthread_t *vdec_thread, hi_u32 send_arr_len, hi_u32 thread_arr_len);
hi_void sample_comm_vdec_start_send_stream(hi_s32 chn_num, vdec_thread_param *vdec_send, pthread_t *vdec_thread,
    hi_u32 send_arr_len, hi_u32 thread_arr_len);
hi_void sample_comm_vdec_stop_send_stream(hi_s32 chn_num, vdec_thread_param *vdec_send, pthread_t *vdec_thread,
    hi_u32 send_arr_len, hi_u32 thread_arr_len);
hi_void *sample_comm_vdec_send_stream(hi_void *args);
hi_s32 sample_comm_vdec_start(hi_s32 chn_num, sample_vdec_attr *past_sample_vdec, hi_u32 arr_len);
hi_s32 sample_comm_vdec_stop(hi_s32 chn_num);

#ifdef __cplusplus
}
#endif /* end of #ifdef __cplusplus */

#endif /* end of #ifndef __SAMPLE_COMMON_H__ */