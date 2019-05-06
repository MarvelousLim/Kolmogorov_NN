import numpy as np, psi, pycuda.driver as cuda, pycuda.autoinit
from pycuda.compiler import SourceModule
gamma = 10
n = 3
limit = 15

def block_grid(size):
    block_size = 1024
    grid_size = size // block_size
    if (size % block_size != 0):
        grid_size += 1
    return block_size, grid_size

class second_layer:
    def __init__(self, N, n):
        mod = SourceModule("""
        
            #define FULL_MASK 0xffffffff
        
            __device__ float warpReduceSum(float val) 
            {
                for (int offset = warpSize >> 1; offset > 0; offset >>= 1)
                    val += __shfl_down_sync(FULL_MASK, val, offset);
                return val;
            }    

            __device__
            float predict_single(float *x, float *theta, int n, float step)
            {
                float pred = 0;
                for (int j = 0; j < 2*n+1; j++)
                    pred += theta[j + int(x[j] / step) * (2*n+1)];
                return pred;
            }

            __global__
            void to_zero(float *gradd, int n, int N, float *Q)
            {
                int i = threadIdx.x + blockIdx.x * blockDim.x;
                if (i < (2*n+1)*N)
                {
                    gradd[i] = 0;
                }
                if (i == 0)
                    Q[0] = 0.0;
            }

            __global__
            void grad(float *X, float *Y, int n_rows, float *theta,
int n, int N, float up_max, float *gradd, float *Q)
            {
                int i = threadIdx.x + blockIdx.x * blockDim.x;
                if (i < n_rows)
                {
                    float step = up_max / N, x, y = Y[i], delta;
                    int pointer = i * (2*n+1);

                    delta = predict_single(X + pointer, theta, n, step) - y;

                    for (int j = 0; j < 2*n+1; j++)
                    {
                        x = X[j + pointer];
                        atomicAdd(gradd + j + int(x / step) * (2*n+1), delta / n_rows);
                    }
                    
                    float reductionRes = warpReduceSum(delta * delta / n_rows);
                    if ((threadIdx.x & (warpSize - 1)) == 0)
                        atomicAdd(Q, reductionRes);
                }
            }
            __global__
            void adam(float *theta, int n, int N, float *gradd, int t, float *m, float *v, float alpha)
            {
                float beta_1 = 0.9;
                float beta_2 = 0.999;
                float epsilon = 1e-8;

                int i = threadIdx.x + blockIdx.x * blockDim.x;
                if (i < (2*n+1)*N)
                {
                    m[i] = beta_1 * m[i] + (1 - beta_1) * gradd[i];
                    v[i] = beta_2 * v[i] + (1 - beta_2) * gradd[i] * gradd[i];
                    float m_hat = m[i] / (1 - powf(beta_1, t));
                    float v_hat = v[i] / (1 - powf(beta_2, t));

                    theta[i] -= alpha * m_hat / (epsilon + sqrtf(v_hat));
                }
            }

            __global__
            void predict(float *X, float *Y, int n_rows, float *theta, int n, float step)
            {
                int i = threadIdx.x + blockIdx.x * blockDim.x;
                if (i < n_rows)
                {
                    int pointer = i * (2*n+1);
                    Y[i] = predict_single(X + pointer, theta, n, step);
                }
            }

            """)
        #fucntions
        self.grad = mod.get_function("grad")
        self.to_zero = mod.get_function("to_zero")
        self.adam = mod.get_function("adam")
        self.predict_cuda = mod.get_function("predict")
        
        #
        self.N = N
        self.n = n
        self.theta = np.zeros(N *(2*n+1), dtype=np.float32)
        self.up_max = np.array(sum([psi.lbd(p + 1, gamma, n) for p in range(n)]), dtype=np.float32)
        self.loss_memory = []
        self.t = 1
        self.m = np.zeros_like(self.theta)
        self.v = np.zeros_like(self.theta)

    def predict(self, X):        
        Y = np.zeros(X.shape[0], dtype=np.float32)
        X = X.flatten().astype(np.float32)
        #donwload parameters
        theta_gpu = cuda.mem_alloc(self.theta.nbytes)
        cuda.memcpy_htod(theta_gpu, self.theta)
        #download data
        X_gpu = cuda.mem_alloc(X.nbytes)
        cuda.memcpy_htod(X_gpu, X)
        Y_gpu = cuda.mem_alloc(Y.nbytes)
        #fire!
        block_size, grid_size = block_grid(Y.shape[0])
        self.predict_cuda(X_gpu, Y_gpu, np.int32(Y.shape[0]), theta_gpu, np.int32(n), np.float32(self.up_max / self.N),
               block=(block_size, 1, 1), grid=(grid_size, 1))
        #upload result
        cuda.memcpy_dtoh(Y, Y_gpu)
        #release data and params
        X_gpu.free()
        Y_gpu.free()
        theta_gpu.free()
        return Y
    def fit(self, X, Y, alpha=0.1, n_steps=10):
        #donwload parameters
        theta_gpu = cuda.mem_alloc(self.theta.nbytes)
        cuda.memcpy_htod(theta_gpu, self.theta)
        #download data
        X_gpu = cuda.mem_alloc(X_train.nbytes)
        Y_gpu = cuda.mem_alloc(Y_train.nbytes)
        cuda.memcpy_htod(X_gpu, X_train.flatten().astype(np.float32))
        cuda.memcpy_htod(Y_gpu, Y_train.astype(np.float32))
        #init and download additional temp parametes
        gradd = np.zeros_like(self.theta)
        Q = np.zeros_like(self.up_max)
        
        gradd_gpu = cuda.mem_alloc(self.theta.nbytes)
        m_gpu = cuda.mem_alloc(self.m.nbytes)
        v_gpu = cuda.mem_alloc(self.v.nbytes)
        Q_gpu = cuda.mem_alloc(Q.nbytes)
        
        cuda.memcpy_htod(m_gpu, self.m)
        cuda.memcpy_htod(v_gpu, self.v)
        #fire!
        for p in log_progress(range(n_steps), name="step"):
            block_size, grid_size = block_grid(self.theta.shape[0])    
            self.to_zero(gradd_gpu, np.int32(n), np.int32(N), Q_gpu, 
                    block=(block_size, 1, 1), grid=(grid_size, 1))

            block_size, grid_size = block_grid(Y_train.shape[0])    
            self.grad(X_gpu, Y_gpu, np.int32(Y_train.shape[0]), theta_gpu, np.int32(n), 
                 np.int32(N), self.up_max, np.int32(10), gradd_gpu, Q_gpu, 
                 block=(block_size, 1, 1), grid=(grid_size, 1))

            block_size, grid_size = block_grid(self.theta.shape[0])    
            self.adam(theta_gpu, np.int32(n), np.int32(N), gradd_gpu, np.int32(self.t), m_gpu, v_gpu, np.float32(alpha),
                block=(block_size, 1, 1), grid=(grid_size, 1))
            cuda.memcpy_dtoh(Q, Q_gpu)
            self.loss_memory.append(Q.tolist())
            self.t += 1
        #update params
        cuda.memcpy_dtoh(self.theta, theta_gpu)
        cuda.memcpy_dtoh(self.m, m_gpu)
        cuda.memcpy_dtoh(self.v, v_gpu)
        #release data and params
        X_gpu.free()
        Y_gpu.free()
        theta_gpu.free()
        gradd_gpu.free()
        m_gpu.free()
        v_gpu.free()
        Q_gpu.free()