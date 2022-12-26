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
// #include "NvCudaProc.h"
#include "nvbuf_utils.h"

#include "camera_v4l2_cuda.h"

// CUDA as Producer

#define WIDTH   1920
#define HEIGHT  1080

void ProducerThread(EGLStreamKHR eglStream) 
{
    //Prepares frame
    CUeglFrame* cudaEgl = (CUeglFrame *)malloc(sizeof(CUeglFrame));
    cudaEgl->width = WIDTH;
    cudaEgl->depth = 0;
    cudaEgl->planeDesc[0].height = HEIGHT;
    cudaEgl->planeDesc[0].numChannels = 4;
    cudaEgl->planeDesc[0].pitch = WIDTH * cudaEgl->planeDesc[0].numChannels;
    cudaEgl->frameType = cudaEglFrameTypePitch;
    cudaEgl->planeCount = 1;
    cudaEgl->eglColorFormat = cudaEglColorFormatARGB;
    cudaEgl->planeDesc[0].channelDesc.f=cudaChannelFormatKindUnsigned
    cudaEgl->planeDesc[0].channelDesc.w = 8;
    cudaEgl->planeDesc[0].channelDesc.x = 8;
    cudaEgl->planeDesc[0].channelDesc.y = 8;
    cudaEgl->planeDesc[0].channelDesc.z = 8;
    size_t numElem = cudaEgl->planeDesc[0].pitch * cudaEgl->planeDesc[0].height;
    // Buffer allocated by producer
    cudaMalloc(&(cudaEgl->pPitch[0].ptr), numElem);
    // CUDA producer connects to EGLStream
    cudaEGLStreamProducerConnect(&conn, eglStream, WIDTH, HEIGHT);
    // Sets all elements in the buffer to 1
    K1<<<...>>>(cudaEgl->pPitch[0].ptr, 1, numElem);
    // Present frame to EGLStream
    cudaEGLStreamProducerPresentFrame(&conn, *cudaEgl, NULL);
    cudaEGLStreamProducerReturnFrame(&conn, cudaEgl, eglStream);
    // .
    // .
    // clean up
    cudaEGLStreamProducerDisconnect(&conn);
    //.
}

// CUDA as Consumer
void ConsumerThread(EGLStreamKHR eglStream) 
{
    // . .
    //Connect consumer to EGLStream
    cudaEGLStreamConsumerConnect(&conn, eglStream);
    // consumer acquires a frame
    unsigned int timeout = 16000;
    cudaEGLStreamConsumerAcquireFrame(& conn, &cudaResource, eglStream, timeout);
    //consumer gets a cuda object pointer
    cudaGraphicsResourceGetMappedEglFrame(&cudaEgl, cudaResource, 0, 0);
    size_t numElem = cudaEgl->planeDesc[0].pitch * cudaEgl->planeDesc[0].height;
    // . .
    int checkIfOne = 1;
    // Checks if each value in the buffer is 1, if any value is not 1, it sets
    checkIfOne = 0.
    K2<<<...>>>(cudaEgl->pPitch[0].ptr, 1, numElem, checkIfOne);
    // . .
    cudaEGLStreamConsumerReleaseFrame(&conn, cudaResource, &eglStream);
    // . .
    cudaEGLStreamConsumerDisconnect(&conn);
    // .
}

// CUDA interop with EGLImage
int width = 256;
int height = 256;

int main()
{
    // .
    // .
    unsigned char *hostSurf;
    unsigned char *pSurf;
    CUarray pArray;
    unsigned int bufferSize = WIDTH * HEIGHT * 4;
    pSurf= (unsigned char *)malloc(bufferSize); 
    hostSurf = (unsigned char*)malloc(bufferSize);
    // Initialize the buffer
    
    for(int y = 0; y < HEIGHT; y++)
    {
        for(int x = 0; x < WIDTH; x++)
        {
            pSurf[(y*WIDTH + x) * 4 ] = 0; pSurf[(y*WIDTH + x) * 4 + 1] = 0;
            pSurf[(y*WIDTH + x) * 4 + 2] = 0; pSurf[(y*WIDTH + x) * 4 + 3] = 0;
        }
    }

    // NOP call to error-check the above glut calls
    GL_SAFE_CALL({});
    //Init texture
    GL_SAFE_CALL(glGenTextures(1, &tex));
    GL_SAFE_CALL(glBindTexture(GL_TEXTURE_2D, tex));
    GL_SAFE_CALL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WIDTH, HEIGHT, 0, GL_RGBA,
    GL_UNSIGNED_BYTE, pSurf));
    EGLDisplay eglDisplayHandle = eglGetCurrentDisplay();
    EGLContext eglCtx = eglGetCurrentContext();
    // Create the EGL_Image
    EGLint eglImgAttrs[] = { EGL_IMAGE_PRESERVED_KHR, EGL_FALSE, EGL_NONE, EGL_NONE };
    EGLImageKHR eglImage = eglCreateImageKHR(eglDisplayHandle, eglCtx, EGL_GL_TEXTURE_2D_KHR, (EGLClientBuffer)(intptr_t)tex, eglImgAttrs);
    glFinish();
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE,
    pSurf);
    glFinish();
    // Register buffer with CUDA
    cuGraphicsEGLRegisterImage(&pResource, eglImage,0);
    // Get CUDA array from graphics resource object
    cuGraphicsSubResourceGetMappedArray( &pArray, pResource, 0, 0);
    cuCtxSynchronize();
    // Create a CUDA surface object from pArray
    CUresult status = CUDA_SUCCESS;
    CUDA_RESOURCE_DESC wdsc;
    memset(&wdsc, 0, sizeof(wdsc));
    wdsc.resType = CU_RESOURCE_TYPE_ARRAY; wdsc.res.array.hArray = pArray;
    CUsurfObject writeSurface;
    cuSurfObjectCreate(&writeSurface, &wdsc);
    dim3 blockSize(32,32);
    dim3 gridSize(width/blockSize.x,height/blockSize.y);
    // Modifies the OpenGL texture using CUDA surface object
    changeTexture<<<gridSize, blockSize>>>(writeSurface, width, height);
    cuCtxSynchronize();
    CUDA_MEMCPY3D cpdesc;
    memset(&cpdesc, 0, sizeof(cpdesc));
    cpdesc.srcXInBytes = cpdesc.srcY = cpdesc.srcZ = cpdesc.srcLOD = 0;
    cpdesc.dstXInBytes = cpdesc.dstY = cpdesc.dstZ = cpdesc.dstLOD = 0;
    cpdesc.srcMemoryType = CU_MEMORYTYPE_ARRAY; 
    cpdesc.dstMemoryType = CU_MEMORYTYPE_HOST;
    cpdesc.srcArray = pArray;
    cpdesc.dstHost = (void *)hostSurf;
    cpdesc.WidthInBytes = WIDTH * 4; 
    cpdesc.Height = HEIGHT; 
    cpdesc.Depth = 1;

    //Copy CUDA surface object values to hostSurf
    cuMemcpy3D(&cpdesc);
    cuCtxSynchronize();
    unsigned char* temp = (unsigned char*)(malloc(bufferSize * sizeof(unsigned char)));
    // Get the modified texture values as
    GL_SAFE_CALL(glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE,
    (void*)temp));
    glFinish();
    // Check if the OpenGL texture got modified values
    checkbuf(temp,hostSurf);
    // Clean up CUDA
    cuGraphicsUnregisterResource(pResource);
    cuSurfObjectDestroy(writeSurface);
    // .
    // .
}

__global__ void 
changeTexture(cudaSurfaceObject_t arr, unsigned int width, unsigned int height)
{
    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
    uchar4 data = make_uchar4(1, 2, 3, 4);
    surf2Dwrite(data, arr, x * 4, y);
}

    
void checkbuf(unsigned char *ref, unsigned char *hostSurf) 
{
    for(int y = 0; y < height*width*4; y++)
    {
        if (ref[y] != hostSurf[y])
            printf("mis match at %d\n",y);
    }
}

// Creating EGLSync from a CUDA Event

EGLDisplay dpy = eglGetCurrentDisplay();
// Create CUDA event
cudaEvent_t event;
cudaStream_t *stream;
cudaEventCreate(&event);
cudaStreamCreate(&stream);
// Record the event with cuda event
cudaEventRecord(event, stream);

const EGLAttrib attribs[] = {
    EGL_CUDA_EVENT_HANDLE_NV, (EGLAttrib )event,
    EGL_NONE
};

//Create EGLSync from the cuda event
eglsync = eglCreateSync(dpy, EGL_NV_CUDA_EVENT_NV, attribs);
//Wait on the sync
eglWaitSyncKHR(...);

// Creating a CUDA Event from EGLSync
EGLSync eglsync;
EGLDisplay dpy = eglGetCurrentDisplay();
// Create an eglSync object from openGL fense sync object
eglsync = eglCreateSyncKHR(dpy, EGL_SYNC_FENCE_KHR, NULL);
cudaEvent_t event;
cudaStream_t* stream;
cudaStreamCreate(&stream);
// Create CUDA event from eglSync
cudaEventCreateFromEGLSync(&event, eglSync, cudaEventDefault);
// Wait on the cuda event. It waits on GPU till OpenGL finishes its
// task
cudaStreamWaitEvent(stream, event, 0);

// ------------------------------------------------------------------------
// the usage of an EGLSync interop.
int width = 256;
int height = 256;
int main()
{
    // .
    // .
    unsigned char *hostSurf;
    unsigned char *pSurf;
    cudaArray_t pArray;
    unsigned int bufferSize = WIDTH * HEIGHT * 4;
    pSurf = (unsigned char *)malloc(bufferSize); 
    hostSurf = (unsigned char*)malloc(bufferSize);
    // Intialize the buffer
    for(int y = 0; y < bufferSize; y++)
        pSurf[y] = 0;

    //Init texture
    GL_SAFE_CALL(glGenTextures(1, &tex));
    GL_SAFE_CALL(glBindTexture(GL_TEXTURE_2D, tex));
    GL_SAFE_CALL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, pSurf));
    
    EGLDisplay eglDisplayHandle = eglGetCurrentDisplay();
    EGLContext eglCtx = eglGetCurrentContext();
    cudaEvent_t cuda_event;
    cudaEventCreateWithFlags(cuda_event, cudaEventDisableTiming);
    EGLAttribKHR eglattrib[] = {
        EGL_CUDA_EVENT_HANDLE_NV, (EGLAttrib) cuda_event, EGL_NONE
        };

    cudaStream_t* stream;
    cudaStreamCreateWithFlags(&stream,cudaStreamDefault);
    EGLSyncKHR eglsync1, eglsync2;
    cudaEvent_t egl_event;
    // Create the EGL_Image
    EGLint eglImgAttrs[] = { EGL_IMAGE_PRESERVED_KHR, EGL_FALSE, EGL_NONE, EGL_NONE };
    EGLImageKHR eglImage = eglCreateImageKHR(eglDisplayHandle, eglCtx,
    EGL_GL_TEXTURE_2D_KHR, (EGLClientBuffer)(intptr_t)tex, eglImgAttrs);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, pSurf);
    //Creates an EGLSync object from GL Sync object to track
    //finishing of copy.
    eglsync1 = eglCreateSyncKHR(eglDisplayHandle, EGL_SYNC_FENCE_KHR, NULL);
    //Create CUDA event object from EGLSync obejct
    cuEventCreateFromEGLSync(&egl_event, eglsync1, cudaEventDefault);
    //Waiting on GPU to finish GL copy
    cuStreamWaitEvent(stream, egl_event, 0);
    // Register buffer with CUDA
    cudaGraphicsEGLRegisterImage(&pResource, eglImage, cudaGraphicsRegisterFlagsNone);
    //Get CUDA array from graphics resource object
    cudaGraphicsSubResourceGetMappedArray( &pArray, pResource, 0, 0);
    // .
    // .
    //Create a CUDA surface object from pArray
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray; resDesc.res.array.array = pArray;
    cudaSurfaceObject_t inputSurfObj = 0;
    cudaCreateSurfaceObject(&inputSurfObj, &resDesc);
    dim3 blockSize(32,32);
    dim3 gridSize(width/blockSize.x,height/blockSize.y);
    // Modifies the CUDA array using CUDA surface object
    changeTexture<<<gridSize, blockSize>>>(inputSurfObj, width, height);
    cuEventRecord(cuda_event, stream);
    //Create EGLsync object from CUDA event cuda_event
    eglsync2 = eglCreateSync64KHR(dpy, EGL_SYNC_CUDA_EVENT_NV, eglattrib);
    //waits till kernel to finish
    eglWaitSyncKHR(eglDisplayHandle, eglsync2, 0);
    // .
    //Copy modified pArray values to hostSurf
    // .
    unsigned char* temp = (unsigned char*)(malloc(bufferSize * sizeof(unsigned char)));
    // Get the modified texture values
    GL_SAFE_CALL(glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, (void*)temp));

    // .
    // .
    // This function check if the OpenGL texture got modified values
    checkbuf(temp,hostSurf);
    // Clean up CUDA
    cudaGraphicsUnregisterResource(pResource);
    cudaDestroySurfaceObject(inputSurfObj);
    eglDestroySyncKHR(eglDisplayHandle, eglsync1);
    eglDestroySyncKHR(eglDisplayHandle, eglsync2);
    cudaEventDestroy(egl_event);
    cudaEventDestroy(cuda_event);
    // .
    // .
}