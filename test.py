import timeit
import numpy as np
import sys

try:
    import numox
except ImportError:
    print("Error: numox module not found. Please compile it first.")
    print("Try running: python setup.py build_ext --inplace")
    sys.exit(1)

# Predefined testing gradients
SIZES = [100, 500, 1000]
ITERATIONS_SMALL = 100
ITERATIONS_LARGE = 10

def benchmark():
    print(f"{'='*60}")
    print(f"{'numox vs numpy Benchmark (Scientific)':^60}")
    print(f"{'='*60}\n")
    
    for size in SIZES:
        iters = ITERATIONS_SMALL if size < 1000 else ITERATIONS_LARGE
        print(f"--- Matrix Size: {size}x{size} | Iterations: {iters} ---")
        
        # 1. Initialize numpy matrices
        np_a = np.random.uniform(0.0, 1.0, (size, size))
        np_b = np.random.uniform(0.0, 1.0, (size, size))
        
        # 2. Initialize numox matrices
        try:
            nx_a = numox.Matrix(size, size, rand=True, low=0.0, high=1.0, seed=42)
            nx_b = numox.Matrix(size, size, rand=True, low=0.0, high=1.0, seed=43)
        except Exception:
            # Fallback just in case the signature is different
            try:
                nx_a = numox.Matrix(np_a.tolist())
                nx_b = numox.Matrix(np_b.tolist())
            except Exception as e:
                print(f"Failed to initialize numox matrices: {e}")
                return
            
        # 3. Addition Test
        print(">> Addition (+)")
        t_np_add = timeit.timeit(lambda: np_a + np_b, number=iters)
        t_nx_add = timeit.timeit(lambda: nx_a + nx_b, number=iters)
        print(f"   NumPy: {t_np_add:.5f} s")
        print(f"   numox: {t_nx_add:.5f} s  (Speedup: {t_np_add/t_nx_add:.2f}x)")
        
        # 4. Subtraction Test
        print(">> Subtraction (-)")
        t_np_sub = timeit.timeit(lambda: np_a - np_b, number=iters)
        t_nx_sub = timeit.timeit(lambda: nx_a - nx_b, number=iters)
        print(f"   NumPy: {t_np_sub:.5f} s")
        print(f"   numox: {t_nx_sub:.5f} s  (Speedup: {t_np_sub/t_nx_sub:.2f}x)")
        
        # 5. Multiplication Test
        # NumPy uses @ (matmul) for true matrix multiplication, * is element-wise.
        # We assume numox overrides * to mean matrix multiplication (as typical in C extensions).
        print(">> Multiplication (*)")
        t_np_mul = timeit.timeit(lambda: np_a @ np_b, number=iters)
        t_nx_mul = timeit.timeit(lambda: nx_a * nx_b, number=iters)
        print(f"   NumPy: {t_np_mul:.5f} s")
        print(f"   numox: {t_nx_mul:.5f} s  (Speedup: {t_np_mul/t_nx_mul:.2f}x)\n")

if __name__ == '__main__':
    benchmark()
