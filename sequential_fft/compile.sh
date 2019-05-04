TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

# Use 0 if the TensorFlow binary is built with GCC 4.x
# see https://docs.computecanada.ca/wiki/GCC_C%2B%2B_Dual_ABI for details
USE_CXX11_ABI=0

nvcc -std=c++11 -c -o sequential_batch_fft_kernel.cu.o \
  sequential_batch_fft_kernel.cu.cc \
  -D_GLIBCXX_USE_CXX11_ABI=$USE_CXX11_ABI -DNDEBUG \
  -L$TF_LIB -ltensorflow_framework \
  -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 -shared -o ./build/sequential_batch_fft.so \
  sequential_batch_fft_kernel.cu.o \
  sequential_batch_fft.cc \
  -D_GLIBCXX_USE_CXX11_ABI=$USE_CXX11_ABI -DNDEBUG \
  -L$TF_LIB -ltensorflow_framework \
  -I $TF_INC -fPIC \
  -lcudart -lcufft -L/usr/local/cuda/lib64

rm -rf sequential_batch_fft_kernel.cu.o
