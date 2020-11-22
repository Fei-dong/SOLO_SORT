import cv2
import time
import logging
logging.basicConfig(level = logging.DEBUG,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import pycuda.autoinit
import pycuda.driver as drv
import numpy

from pycuda.compiler import SourceModule
mod = SourceModule("""
__global__ void func(float *mask, float *a, float *b, size_t w, size_t h)
{
  // const int i = threadIdx.x;
  const int i =  blockIdx.x * blockDim.x + threadIdx.x;
  if(blockIdx.x<3&&threadIdx.x<10){
      //printf("%d\\n",i);
      printf("blockIdx.x = %d; threadIdx.x = %d; blockDim.x = %d; i = %d;\\n", blockIdx.x, threadIdx.x, blockDim.x, i);
  }
  //a[i] = i;
  //if(mask[i] == 1){
  if(mask[i] > 0.5 ){
      a[i] = i % w;
      b[i] = i / w;
      printf("w = %d; a[i] = %f; b[i] = %f; i = %d;\\n", w, a[i], b[i], i);
      printf("mask[i] = %f;\\n", mask[i]);
  }
}
""")

func = mod.get_function("func")

# kernel_code = r"""
# __global__ void print_id(void)
# {
#     printf("blockIdx.x = %d; threadIdx.x = %d;\n", blockIdx.x, threadIdx.x);
# }
# """
# mod = SourceModule(kernel_code)
# print_id = mod.get_function("print_id")
# print_id(grid=(2,), block=(30,1,1))

def test():
    mask = cv2.imread('cur_mask.jpg')
    mask = mask[:,:,0]
    # mask_g = mask[:,:,1]
    # mask_b = mask[:,:,2]

    x = []
    y = []
    prev_time = time.time()
    for i in range(len(mask)):
        for j in range(len(mask[0])):
            if mask[i][j] == 2:
                print('x = %d,y = %d'%(j,i))
                y.append(i)
                x.append(j)
    logger.info('process frame time:'+str(time.time()-prev_time))
    prev_time = time.time()
    x1 = min(x)
    y1 = min(y)
    x2 = max(x)
    y2 = max(y)
    logger.info('post process frame time:'+str(time.time()-prev_time))
    print(x1,y1,x2,y2)


    # w = numpy.int32(len(mask[0]))
    # h = numpy.int32(len(mask))
    w = numpy.int64(len(mask[0]))
    h = numpy.int64(len(mask))
    mask_np = numpy.array(mask)
    mask_np = mask_np.reshape(-1).astype(float)
    N = len(mask_np)
    # print(N)
    # print(w)
    # print(h)
    a = numpy.zeros(N, dtype=numpy.float)
    b = numpy.zeros(N, dtype=numpy.float)
    nTheads = 1024
    nBlocks = int( ( N + nTheads - 1 ) / nTheads )
    print("nBlocks:%d\n"%nBlocks)
    prev_time = time.time()
    func(
            drv.In(mask_np), drv.InOut(a), drv.InOut(b),w,h,
            block=( nTheads, 1, 1 ), grid=( nBlocks,  ) )
    logger.info('gpu process frame time:'+str(time.time()-prev_time))
    print(max(a))
    print(max(b))
    print(a)
    # img_show = cv2.rectangle(mask, (x1,y1), (x2,y2), (0,255,0), 2)
    # cv2.imwrite('spend.jpg',img_show)

if __name__ == '__main__':
    test()