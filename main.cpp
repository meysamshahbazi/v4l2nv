/*
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <stdio.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <errno.h>
#include <stdlib.h>
#include <signal.h>
#include <poll.h>

#include "NvEglRenderer.h"
#include "NvUtils.h"

#include <stdio.h>
#include <cuda_runtime_api.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "cudaEGL.h"
#include "cudaColorspace.h"


#include "cudaDraw.h"
#include "cudaFont.h"
// #include "NvCudaProc.h"
#include "nvbuf_utils.h"

#include "camera_v4l2_cuda.h"

#include "cudaColorspace.h"

#include "cudaRGB.h"
#include "cudaYUV.h"
#include "cudaBayer.h"
#include "cudaGrayscale.h"

#include "logging.h"


#define MJPEG_EOS_SEARCH_SIZE 4096

static bool quit = false;

using namespace std;

uchar4* rgb_img = NULL;
cudaFont* font = NULL; // = cudaFont::Create();
static void
set_defaults(context_t * ctx)
{
    memset(ctx, 0, sizeof(context_t));

    ctx->cam_devname = "/dev/video0";
    ctx->cam_fd = -1;
    ctx->cam_pixfmt = V4L2_PIX_FMT_YUYV;
    ctx->cam_w = 1920;
    ctx->cam_h = 1080;
    ctx->frame = 0;
    // ctx->save_n_frame = 0;

    ctx->g_buff = NULL;
    ctx->capture_dmabuf = true;
    ctx->renderer = NULL;
    ctx->fps = 30;

    ctx->enable_cuda = false;
    ctx->egl_image = NULL;
    ctx->egl_display = EGL_NO_DISPLAY;

    ctx->enable_verbose = false;
    font = cudaFont::Create(40.0f);
    if( !font )
	{
		printf("gl-display-test:  failed to create cudaFont object\n");
	}
    // size_t bufferSize = 1920 * 1080 * sizeof(uchar4);
    // CUresult cuResult = cuMemAlloc((CUdeviceptr*)rgb_img, bufferSize);
    // cudaError_t res;
    // res = cudaMalloc((void **) &rgb_img,bufferSize);
    
}

static nv_color_fmt nvcolor_fmt[] =
{
    /* TODO: add more pixel format mapping */
    {V4L2_PIX_FMT_UYVY, NvBufferColorFormat_UYVY},
    {V4L2_PIX_FMT_VYUY, NvBufferColorFormat_VYUY},
    {V4L2_PIX_FMT_YUYV, NvBufferColorFormat_YUYV},
    {V4L2_PIX_FMT_YVYU, NvBufferColorFormat_YVYU},
    {V4L2_PIX_FMT_GREY, NvBufferColorFormat_GRAY8},
    {V4L2_PIX_FMT_YUV420M, NvBufferColorFormat_YUV420},
};

static NvBufferColorFormat
get_nvbuff_color_fmt(unsigned int v4l2_pixfmt)
{
    unsigned i;

    for (i = 0; i < sizeof(nvcolor_fmt) / sizeof(nvcolor_fmt[0]); i++)
    {
        if (v4l2_pixfmt == nvcolor_fmt[i].v4l2_pixfmt)
            return nvcolor_fmt[i].nvbuff_color;
    }

    return NvBufferColorFormat_Invalid;
}

static bool
camera_initialize(context_t * ctx)
{
    struct v4l2_format fmt;

    /* Open camera device */
    ctx->cam_fd = open(ctx->cam_devname, O_RDWR);
    if (ctx->cam_fd == -1)
        ERROR_RETURN("Failed to open camera device %s: %s (%d)",
                ctx->cam_devname, strerror(errno), errno);

    /* Set camera output format */
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = ctx->cam_w;
    fmt.fmt.pix.height = ctx->cam_h;
    fmt.fmt.pix.pixelformat = ctx->cam_pixfmt;
    fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;// V4L2_FIELD_INTERLACED
    if (ioctl(ctx->cam_fd, VIDIOC_S_FMT, &fmt) < 0)
        ERROR_RETURN("Failed to set camera output format: %s (%d)",
                strerror(errno), errno);

    /* Get the real format in case the desired is not supported */
    memset(&fmt, 0, sizeof fmt);
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(ctx->cam_fd, VIDIOC_G_FMT, &fmt) < 0)
        ERROR_RETURN("Failed to get camera output format: %s (%d)",
                strerror(errno), errno);
    if (fmt.fmt.pix.width != ctx->cam_w ||
            fmt.fmt.pix.height != ctx->cam_h ||
            fmt.fmt.pix.pixelformat != ctx->cam_pixfmt)
    {
        WARN("The desired format is not supported");
        ctx->cam_w = fmt.fmt.pix.width;
        ctx->cam_h = fmt.fmt.pix.height;
        ctx->cam_pixfmt =fmt.fmt.pix.pixelformat;
    }

    struct v4l2_streamparm streamparm;
    memset (&streamparm, 0x00, sizeof (struct v4l2_streamparm));
    streamparm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl (ctx->cam_fd, VIDIOC_G_PARM, &streamparm);

    INFO("Camera ouput format: (%d x %d)  stride: %d, imagesize: %d, frate: %u / %u",
            fmt.fmt.pix.width,
            fmt.fmt.pix.height,
            fmt.fmt.pix.bytesperline,
            fmt.fmt.pix.sizeimage,
            streamparm.parm.capture.timeperframe.denominator,
            streamparm.parm.capture.timeperframe.numerator);

    return true;
}

static bool
display_initialize(context_t * ctx)
{
    /* Create EGL renderer */
    ctx->renderer = NvEglRenderer::createEglRenderer("renderer0",
            ctx->cam_w, ctx->cam_h, 0, 0);
    if (!ctx->renderer)
        ERROR_RETURN("Failed to create EGL renderer");
    ctx->renderer->setFPS(ctx->fps);

    if (ctx->enable_cuda)
    {
        /* Get defalut EGL display */
        ctx->egl_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if (ctx->egl_display == EGL_NO_DISPLAY)
            ERROR_RETURN("Failed to get EGL display connection");

        /* Init EGL display connection */
        if (!eglInitialize(ctx->egl_display, NULL, NULL))
            ERROR_RETURN("Failed to initialize EGL display connection");
    }

    return true;
}

static bool
init_components(context_t * ctx)
{
    if (!camera_initialize(ctx))
        ERROR_RETURN("Failed to initialize camera device");

    if (!display_initialize(ctx))
        ERROR_RETURN("Failed to initialize display");

    INFO("Initialize v4l2 components successfully");
    return true;
}

static bool
request_camera_buff(context_t *ctx)
{
    /* Request camera v4l2 buffer */
    printf("****request_camera_buff****\n");
    struct v4l2_requestbuffers rb;
    memset(&rb, 0, sizeof(rb));
    rb.count = V4L2_BUFFERS_NUM;
    rb.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    rb.memory = V4L2_MEMORY_DMABUF;
    if (ioctl(ctx->cam_fd, VIDIOC_REQBUFS, &rb) < 0)
        ERROR_RETURN("Failed to request v4l2 buffers: %s (%d)",
                strerror(errno), errno);
    if (rb.count != V4L2_BUFFERS_NUM)
        ERROR_RETURN("V4l2 buffer number is not as desired");

    for (unsigned int index = 0; index < V4L2_BUFFERS_NUM; index++)
    {
        struct v4l2_buffer buf;

        /* Query camera v4l2 buf length */
        memset(&buf, 0, sizeof buf);
        buf.index = index;
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_DMABUF;

        if (ioctl(ctx->cam_fd, VIDIOC_QUERYBUF, &buf) < 0)
            ERROR_RETURN("Failed to query buff: %s (%d)",
                    strerror(errno), errno);

        /* TODO: add support for multi-planer
           Enqueue empty v4l2 buff into camera capture plane */
        buf.m.fd = (unsigned long)ctx->g_buff[index].dmabuff_fd;
        if (buf.length != ctx->g_buff[index].size)
        {
            WARN("Camera v4l2 buf length is not expected");
            ctx->g_buff[index].size = buf.length;
        }

        if (ioctl(ctx->cam_fd, VIDIOC_QBUF, &buf) < 0)
            ERROR_RETURN("Failed to enqueue buffers: %s (%d)",
                    strerror(errno), errno);
    }

    return true;
}

static bool
request_camera_buff_mmap(context_t *ctx)
{
    /* Request camera v4l2 buffer */
    struct v4l2_requestbuffers rb;
    memset(&rb, 0, sizeof(rb));
    rb.count = V4L2_BUFFERS_NUM;
    rb.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    rb.memory = V4L2_MEMORY_MMAP;
    if (ioctl(ctx->cam_fd, VIDIOC_REQBUFS, &rb) < 0)
        ERROR_RETURN("Failed to request v4l2 buffers: %s (%d)",
                strerror(errno), errno);
    if (rb.count != V4L2_BUFFERS_NUM)
        ERROR_RETURN("V4l2 buffer number is not as desired");

    for (unsigned int index = 0; index < V4L2_BUFFERS_NUM; index++)
    {
        struct v4l2_buffer buf;

        /* Query camera v4l2 buf length */
        memset(&buf, 0, sizeof buf);
        buf.index = index;
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

        buf.memory = V4L2_MEMORY_MMAP;
        if (ioctl(ctx->cam_fd, VIDIOC_QUERYBUF, &buf) < 0)
            ERROR_RETURN("Failed to query buff: %s (%d)",
                    strerror(errno), errno);

        ctx->g_buff[index].size = buf.length;
        ctx->g_buff[index].start = (unsigned char *)
            mmap (NULL /* start anywhere */,
                    buf.length,
                    PROT_READ | PROT_WRITE /* required */,
                    MAP_SHARED /* recommended */,
                    ctx->cam_fd, buf.m.offset);
        if (MAP_FAILED == ctx->g_buff[index].start)
            ERROR_RETURN("Failed to map buffers");

        if (ioctl(ctx->cam_fd, VIDIOC_QBUF, &buf) < 0)
            ERROR_RETURN("Failed to enqueue buffers: %s (%d)",
                    strerror(errno), errno);
    }

    return true;
}



static bool
prepare_buffers(context_t * ctx)
{
    NvBufferCreateParams input_params = {0};

    /* Allocate global buffer context */
    ctx->g_buff = (nv_buffer *)malloc(V4L2_BUFFERS_NUM * sizeof(nv_buffer));
    if (ctx->g_buff == NULL)
        ERROR_RETURN("Failed to allocate global buffer context");

    input_params.payloadType = NvBufferPayload_SurfArray;
    input_params.width = ctx->cam_w;
    input_params.height = ctx->cam_h;
    input_params.layout = NvBufferLayout_Pitch;

    /* Create buffer and provide it with camera */
    for (unsigned int index = 0; index < V4L2_BUFFERS_NUM; index++)
    {
        int fd;
        NvBufferParams params = {0};

        input_params.colorFormat = get_nvbuff_color_fmt(ctx->cam_pixfmt);
        input_params.nvbuf_tag = NvBufferTag_CAMERA;
        if (-1 == NvBufferCreateEx(&fd, &input_params))
            ERROR_RETURN("Failed to create NvBuffer");

        ctx->g_buff[index].dmabuff_fd = fd;

        if (-1 == NvBufferGetParams(fd, &params))
            ERROR_RETURN("Failed to get NvBuffer parameters");

        if (ctx->capture_dmabuf) {
            if (-1 == NvBufferMemMap(ctx->g_buff[index].dmabuff_fd, 0, NvBufferMem_Read_Write,
                        (void**)&ctx->g_buff[index].start))
                ERROR_RETURN("Failed to map buffer");
        }
    }

    input_params.colorFormat = get_nvbuff_color_fmt(V4L2_PIX_FMT_YUV420M);// 
    input_params.nvbuf_tag = NvBufferTag_NONE;
    /* Create Render buffer */
    if (-1 == NvBufferCreateEx(&ctx->render_dmabuf_fd, &input_params))
        ERROR_RETURN("Failed to create NvBuffer");

    if (ctx->capture_dmabuf) {
        if (!request_camera_buff(ctx))
            ERROR_RETURN("Failed to set up camera buff");
    } else {
        if (!request_camera_buff_mmap(ctx))
            ERROR_RETURN("Failed to set up camera buff");
    }

    INFO("Succeed in preparing stream buffers");
    return true;
}

static bool
start_stream(context_t * ctx)
{
    enum v4l2_buf_type type;

    /* Start v4l2 streaming */
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(ctx->cam_fd, VIDIOC_STREAMON, &type) < 0)
        ERROR_RETURN("Failed to start streaming: %s (%d)",
                strerror(errno), errno);

    usleep(200);

    INFO("Camera video streaming on ...");
    return true;
}

static void
signal_handle(int signum)
{
    printf("Quit due to exit command from user!\n");
    quit = true;
}

static bool
cuda_postprocess(context_t *ctx, int fd)
{
    // this fucntion is for facilating the grabbing pointer to image in device for furtur computions

    /* Create EGLImage from dmabuf fd */
    ctx->egl_image = NvEGLImageFromFd(ctx->egl_display, fd);
    if (ctx->egl_image == NULL)
        ERROR_RETURN("Failed to map dmabuf fd (0x%X) to EGLImage",
                ctx->render_dmabuf_fd);
    
    EGLImageKHR image = ctx->egl_image;
    // codes for HandleEGLImage:
    CUresult status;
    CUeglFrame eglFrame;
    CUgraphicsResource pResource = NULL;

    cudaFree(0);

    status = cuGraphicsEGLRegisterImage(&pResource, image,
                CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
    if (status != CUDA_SUCCESS)
    {
        printf("cuGraphicsEGLRegisterImage failed: %d, cuda process stop\n",
                        status);
        return false;
    }

    status = cuGraphicsResourceGetMappedEglFrame(&eglFrame, pResource, 0, 0);
    if (status != CUDA_SUCCESS)
    {
        printf("cuGraphicsSubResourceGetMappedArray failed\n");
    }

    // status = cuCtxSynchronize();
    // if (status != CUDA_SUCCESS)
    // {
    //     printf("cuCtxSynchronize failed\n");
    // }
    // cuCtxCreate
    
//     typedef struct CUeglFrame_st {
//     union {
//         CUarray pArray[MAX_PLANES];     /**< Array of CUarray corresponding to each plane*/
//         void*   pPitch[MAX_PLANES];     /**< Array of Pointers corresponding to each plane*/
//     } frame;
//     unsigned int width;                 /**< Width of first plane */
//     unsigned int height;                /**< Height of first plane */
//     unsigned int depth;                 /**< Depth of first plane */
//     unsigned int pitch;                 /**< Pitch of first plane */
//     unsigned int planeCount;            /**< Number of planes */
//     unsigned int numChannels;           /**< Number of channels for the plane */
//     CUeglFrameType frameType;           /**< Array or Pitch */
//     CUeglColorFormat eglColorFormat;    /**< CUDA EGL Color Format*/
//     CUarray_format cuFormat;            /**< CUDA Array Format*/
// } CUeglFrame;
    
    // printf("width: %d\t",eglFrame.width);
    // printf("height: %d\t",eglFrame.height);
    // printf("depth: %d\t",eglFrame.depth);
    // printf("pitch: %d\t",eglFrame.pitch);
    // printf("planeCount: %d\t",eglFrame.planeCount);
    // printf("numChannels: %d\t",eglFrame.numChannels);
    // printf("frameType: %d\t",eglFrame.frameType);
    // printf("eglColorFormat: %d\n",eglFrame.eglColorFormat);

    // void * pDevPtr = eglFrame.frame.pPitch[0];
    
    // uchar4* rgb_img = NULL; 
    // size_t bufferSize = 1920 * 1080 * sizeof(uchar4);
    // CUresult cuResult = cuMemAlloc((CUdeviceptr*)rgb_img, bufferSize);

    cudaError_t res;

    res = cudaDrawCircleOnYUYU((void*)eglFrame.frame.pPitch[0], 1920, 1080,
                    960, 540, 540, make_float4(0,255,0,200) );
    // res = cudaMalloc((void **) &rgb_img,bufferSize);

    // if (res != cudaSuccess)
    // {
    //     printf("Failed to allocate CUDA buffer\n");
    // }

    // *img_ptr = (void *) pDevPtr;


    // res = cudaConvertColor( (void*) pDevPtr,IMAGE_YUYV,
	// 				     (void*) rgb_img, IMAGE_RGBA8,
	// 				     1920, 1080,
	// 					 make_float2(0,255) ) ;

    // if (res != cudaSuccess)
    // {
    //     printf("cudaConvertColor failed: %d\n", res);
    // }

    // cudaYVYUToRGB(pDevPtr, (uchar3*)rgb_img, 1920, 1080);

    // status = cuCtxSynchronize();
    // if (status != CUDA_SUCCESS)
    // {
    //     printf("cuCtxSynchronize1 failed after memcpy\n");
    // }

    // cudaDrawCircleOnY( (void*) pDevPtr, pDevPtr, 1920, 1080, IMAGE_RGBA8, 
	// 						100, 100, 50, make_float4(0,255,127,200) ) ;
    // cudaDrawCircleOnYUV420( (void*)eglFrame.frame.pPitch[0], (void*)eglFrame.frame.pPitch[1],(void*)eglFrame.frame.pPitch[2], 1920, 1080, IMAGE_RGBA8, 
	// 						960, 540, 100, make_float4(0,255,0,200) );

    // char str[256];
	// sprintf(str, "AaBbCcDdEeFfGgHhIiJjKkLlMmNn123456890");

    // sprintf(str, "AaBbCcDdEeFf");
    // str = "im here";

	// font->OverlayTextOnPlane((void*)eglFrame.frame.pPitch[0], 1920, 1080,
	// 	str, 200, 200, make_float4(0.0f, 190.0f, 255.0f, 255.0f));
        
    // cudaConvertColor( (void*) rgb_img,IMAGE_RGB8,
	// 				     (void*) pDevPtr, IMAGE_YUYV,
	// 				     1920, 1080,
	// 					 make_float2(0,255) ) ;


    // eglFrame.frame.pPitch[0] = (void*) rgb_img;
    // --------------------------------------------------------------------------
    // TODO:: add cuple of next lines after proccessing ...
    
    status = cuCtxSynchronize();
    if (status != CUDA_SUCCESS)
    {
        printf("cuCtxSynchronize2 failed after memcpy\n");
    }

    status = cuGraphicsUnregisterResource(pResource);
    if (status != CUDA_SUCCESS)
    {
        printf("cuGraphicsEGLUnRegisterResource failed: %d\n", status);
    }


    NvDestroyEGLImage(ctx->egl_display, ctx->egl_image);
    ctx->egl_image = NULL;
   // --------------------------------------------------------------------------
    return true;
}

static bool
start_capture(context_t * ctx)
{
    struct sigaction sig_action;
    struct pollfd fds[1];
    NvBufferTransformParams transParams;

    /* Register a shuwdown handler to ensure
       a clean shutdown if user types <ctrl+c> */
    sig_action.sa_handler = signal_handle;
    sigemptyset(&sig_action.sa_mask);
    sig_action.sa_flags = 0;
    sigaction(SIGINT, &sig_action, NULL);

    

    /* Init the NvBufferTransformParams */
    memset(&transParams, 0, sizeof(transParams));
    transParams.transform_flag = NVBUFFER_TRANSFORM_FILTER;
    transParams.transform_filter = NvBufferTransform_Filter_Smart;

    /* Enable render profiling information */
    ctx->renderer->enableProfiling();

    fds[0].fd = ctx->cam_fd;
    fds[0].events = POLLIN;
    /* Wait for camera event with timeout = 5000 ms */
    while (poll(fds, 1, 5000) > 0 && !quit)
    {
        if (fds[0].revents & POLLIN) {
            struct v4l2_buffer v4l2_buf;

            /* Dequeue a camera buff */
            memset(&v4l2_buf, 0, sizeof(v4l2_buf));
            v4l2_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            if (ctx->capture_dmabuf)
                v4l2_buf.memory = V4L2_MEMORY_DMABUF;
            else
                v4l2_buf.memory = V4L2_MEMORY_MMAP;
            if (ioctl(ctx->cam_fd, VIDIOC_DQBUF, &v4l2_buf) < 0)
                ERROR_RETURN("Failed to dequeue camera buff: %s (%d)",
                        strerror(errno), errno);

            ctx->frame++;

            if (ctx->capture_dmabuf) {
                /* Cache sync for VIC operation since the data is from CPU */
                NvBufferMemSyncForDevice(ctx->g_buff[v4l2_buf.index].dmabuff_fd, 0,
                        (void**)&ctx->g_buff[v4l2_buf.index].start);
            } else {
                /* Copies raw buffer plane contents to an NvBuffer plane */
                Raw2NvBuffer(ctx->g_buff[v4l2_buf.index].start, 0,
                            ctx->cam_w, ctx->cam_h, ctx->g_buff[v4l2_buf.index].dmabuff_fd);
            }

            // place for cuda pre process!
            CUresult  status;
            // CUresult  status = cuCtxSynchronize();
            // if (status != CUDA_SUCCESS)
            // {
            //     printf("cuCtxSynchronize2 failed after memcpy \n");
            // }
            // int ret = 0;
            // ret = NvBufferMemMap (ctx->g_buff[v4l2_buf.index].dmabuff_fd, 0, NvBufferMem_Write, (void**)&ctx->g_buff[v4l2_buf.index].start);
            // NvBufferMemMap (  ctx->g_buff[v4l2_buf.index].dmabuff_fd, 0,
            //                     (void**)&ctx->g_buff[v4l2_buf.index].start );

            // printf("ret %d\n",ret);

            // int ret = 0;
            // NvBufferParams dest_param;
            // ret = NvBufferGetParams (ctx->g_buff[v4l2_buf.index].dmabuff_fd, &dest_param );
            // if (ret ==0)
            // {
            //     printf("height: %u\t",dest_param.height[0]);
            //     printf("layout: %u\t",dest_param.layout[0]);
            //     printf("memsize: %d\t",dest_param.memsize);
            //     printf("num_planes: %u\t",dest_param.num_planes);
            //     printf("nv_buffer_size: %u\t",dest_param.nv_buffer_size);
            //     printf("offset: %u\t",dest_param.offset[0]);
            //     printf("payloadType: %u\t",dest_param.payloadType);
            //     printf("pitch: %u\t",dest_param.pitch[0]);
                // printf("pixel_format: %u\t",dest_param.pixel_format);
            //     printf("psize: %u\t",dest_param.psize[0]);
            //     printf("width: %u\t",dest_param.width[0]);
            //     printf("nv_buffer_size: %u\t",dest_param.nv_buffer_size);
            //     printf("\n");
            // }

            // cudaError_t res = cudaDrawCircleOnYUYU( (void*) ctx->g_buff[v4l2_buf.index].start, 1920, 1080,
			// 				960, 540, 100, make_float4(0,255,0,200) );

            // height: 1080	layout: 0	memsize: 0	num_planes: 1	nv_buffer_size: 1008	offset: 0	payloadType: 0	pitch: 3840	pixel_format: 13	psize: 4194304	width: 1920	nv_buffer_size: 1008
            // width: 1920	height: 1080	depth: 0	pitch: 3840	planeCount: 1	numChannels: 1	frameType: 1	eglColorFormat: 12

            // cudaError_t res1 = cudaDrawCircleOnYUYU( (void*) dest_param.nv_buffer, 1920, 1080,
			// 				960, 540, 100, make_float4(0,255,0,200) );
            
            // cudaDeviceSynchronize();

            // status = cuCtxSynchronize();
            // if (status != CUDA_SUCCESS)
            // {
            //     printf("cuCtxSynchronize2 failed after memcpy %d\n",res);
            // }


            cuda_postprocess(ctx, ctx->g_buff[v4l2_buf.index].dmabuff_fd);
            
            /*  Convert the camera buffer from YUV422 to YUV420P */
            if (-1 == NvBufferTransform(ctx->g_buff[v4l2_buf.index].dmabuff_fd, ctx->render_dmabuf_fd,
                        &transParams))
                ERROR_RETURN("Failed to convert the buffer");



            // cuda_postprocess(ctx, ctx->render_dmabuf_fd);

            /* Preview */
            ctx->renderer->render(ctx->render_dmabuf_fd);
            // char my_str[] = "im here!";
            // ctx->renderer->setOverlayText(my_str, 500, 500);

            /* Enqueue camera buffer back to driver */
            if (ioctl(ctx->cam_fd, VIDIOC_QBUF, &v4l2_buf))
                ERROR_RETURN("Failed to queue camera buffers: %s (%d)",
                        strerror(errno), errno);
        }
    }

    /* Print profiling information when streaming stops */
    ctx->renderer->printProfilingStats();



    return true;
}

static bool
stop_stream(context_t * ctx)
{
    enum v4l2_buf_type type;

    /* Stop v4l2 streaming */
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(ctx->cam_fd, VIDIOC_STREAMOFF, &type))
        ERROR_RETURN("Failed to stop streaming: %s (%d)",
                strerror(errno), errno);

    INFO("Camera video streaming off ...");
    return true;
}

int
main(int argc, char *argv[])
{
    context_t ctx;
    int error = 0;

    set_defaults(&ctx);

    // CHECK_ERROR(parse_cmdline(&ctx, argc, argv), cleanup,
    //         "Invalid options specified");

    /* Initialize camera and EGL display, EGL Display will be used to map
       the buffer to CUDA buffer for CUDA processing */
    CHECK_ERROR(init_components(&ctx), cleanup,
            "Failed to initialize v4l2 components");

    // if (ctx.cam_pixfmt == V4L2_PIX_FMT_MJPEG) {
    //     CHECK_ERROR(prepare_buffers_mjpeg(&ctx), cleanup,
    //             "Failed to prepare v4l2 buffs");
    // } else {
    CHECK_ERROR(prepare_buffers(&ctx), cleanup,
            "Failed to prepare v4l2 buffs");
    // }

    CHECK_ERROR(start_stream(&ctx), cleanup,
            "Failed to start streaming");

    CHECK_ERROR(start_capture(&ctx), cleanup,
            "Failed to start capturing")

    CHECK_ERROR(stop_stream(&ctx), cleanup,
            "Failed to stop streaming");

cleanup:
    if (ctx.cam_fd > 0)
        close(ctx.cam_fd);

    if (ctx.renderer != NULL)
        delete ctx.renderer;

    if (ctx.egl_display && !eglTerminate(ctx.egl_display))
        printf("Failed to terminate EGL display connection\n");

    if (ctx.g_buff != NULL)
    {
        for (unsigned i = 0; i < V4L2_BUFFERS_NUM; i++) {
            if (ctx.g_buff[i].dmabuff_fd)
                NvBufferDestroy(ctx.g_buff[i].dmabuff_fd);
            // if (ctx.cam_pixfmt == V4L2_PIX_FMT_MJPEG)
            //     munmap(ctx.g_buff[i].start, ctx.g_buff[i].size);
        }
        free(ctx.g_buff);
    }

    NvBufferDestroy(ctx.render_dmabuf_fd);

    if (error)
        printf("App run failed\n");
    else
        printf("App run was successful\n");

    return -error;
}

