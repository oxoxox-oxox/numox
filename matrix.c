#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200112L
#endif

#include "matrix.h"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#ifdef _WIN32
#include <malloc.h>
#endif

// Include SIMD intrinsics on x86/x86_64
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

static double *alloc_aligned_doubles(size_t count)
{
    size_t bytes = count * sizeof(double);
    if (bytes == 0)
    {
        return NULL;
    }
#ifdef _WIN32
    return (double *)_aligned_malloc(bytes, 32);
#else
    void *ptr = NULL;
    if (posix_memalign(&ptr, 32, bytes) != 0)
    {
        return NULL;
    }
    return (double *)ptr;
#endif
}

static void free_aligned_doubles(double *ptr)
{
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

static inline uint64_t splitmix64(uint64_t x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static inline uint64_t xorshift64star(uint64_t *state)
{
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

/*
 * Generates a random double between `low` and `high`.
 */
double rand_double(double low, double high)
{
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/*
 * Generates a random matrix with `seed` (thread-safe parallel PRNG).
 */
void rand_matrix(matrix *result, unsigned int seed, double low, double high)
{
    if (!result)
    {
        return;
    }
    int rows = result->rows;
    int cols = result->cols;
    double range = high - low;
    const double inv_2_53 = 1.0 / 9007199254740992.0; /* 2^-53 */

    int i;
#pragma omp parallel for if ((size_t)rows * (size_t)cols > 10000)
    for (i = 0; i < rows; i++)
    {
        uint64_t state = splitmix64(((uint64_t)seed << 32) ^ (uint64_t)(i + 1));
        if (state == 0)
        {
            state = 0x9e3779b97f4a7c15ULL;
        }
        double *row_ptr = &result->data[(size_t)i * (size_t)result->stride];
        for (int j = 0; j < cols; j++)
        {
            uint64_t r = xorshift64star(&state);
            double u = (double)(r >> 11) * inv_2_53;
            row_ptr[j] = low + u * range;
        }
    }
}

/*
 *Allocate space for matrix, init all the element in the matrix to 0
 */
int allocate_matrix_raw(matrix **mat, int rows, int cols)
{
    if (mat == NULL || rows <= 0 || cols <= 0)
    {
        return -1;
    }

    size_t r = (size_t)rows;
    size_t c = (size_t)cols;
    if (r > SIZE_MAX / sizeof(double) / c)
    {
        return -1;
    }

    matrix *m = malloc(sizeof(matrix));
    if (m == NULL)
    {
        return -1;
    }
    m->rows = rows;
    m->cols = cols;
    m->is_1d = rows == 1 || cols == 1;
    m->ref_cnt = 1;
    m->data = alloc_aligned_doubles(r * c);
    m->parent = NULL;
    m->stride = cols;

    if (m->data == NULL)
    {
        free(m);
        return -1;
    }

    *mat = m;

    return 0;
}

int allocate_matrix(matrix **mat, int rows, int cols)
{
    int ret = allocate_matrix_raw(mat, rows, cols);

    if (ret == 0)
    {
        memset((*mat)->data, 0, (size_t)rows * (size_t)cols * sizeof(double));
    }
    return ret;
}

/*
 *allocate space for ref of one existing matrix
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int row_offset, int col_offset,
                        int rows, int cols)
{
    if (mat == NULL || from == NULL || rows < 1 || cols < 1 ||
        row_offset < 0 || col_offset < 0 ||
        row_offset > from->rows - rows || col_offset > from->cols - cols)
    {
        return -1;
    }

    matrix *m = malloc(sizeof(matrix));
    if (m == NULL)
    {
        return -1;
    }

    m->rows = rows;
    m->cols = cols;
    m->is_1d = rows == 1 || cols == 1;
    m->ref_cnt = 1;
    m->parent = from;
    m->stride = from->stride;

    m->data = from->data + ((size_t)row_offset * (size_t)from->stride + (size_t)col_offset);

    from->ref_cnt += 1;

    *mat = m;
    return 0;
}

/*
 *deallocate the matrix, also allocate the father of the matrix if it exist
 */
void deallocate_matrix(matrix *mat)
{
    if (mat == NULL)
    {
        return;
    }

    mat->ref_cnt -= 1;

    if (mat->ref_cnt > 0)
    {
        return;
    }

    if (mat->parent == NULL)
    {
        free_aligned_doubles(mat->data);
    }
    else
    {
        deallocate_matrix(mat->parent);
    }
    free(mat);
}

/*
 * Return the double value of the matrix at the given row and column.
 */
double get(matrix *mat, int row, int col)
{
    return mat->data[row * mat->stride + col];
}

/*
 * Set the value at the given row and column to val.
 */
void set(matrix *mat, int row, int col, double val)
{
    mat->data[row * mat->stride + col] = val;
}

/*
 * Set all entries in mat to val
 */
void fill_matrix(matrix *mat, double val)
{
    if (!mat)
    {
        return;
    }
    int r = mat->rows;
    int c = mat->cols;
    size_t total = (size_t)r * (size_t)c;

    if (mat->stride == c)
    {
        double *data = mat->data;
#ifdef __AVX2__
        __m256d v = _mm256_set1_pd(val);
        size_t vec_end = total & ~(size_t)3;
        ptrdiff_t idx;
#pragma omp parallel for if (total > 20000)
        for (idx = 0; idx < (ptrdiff_t)vec_end; idx += 4)
        {
            _mm256_storeu_pd(data + idx, v);
        }
        for (size_t j = vec_end; j < total; j++)
        {
            data[j] = val;
        }
#else
        ptrdiff_t idx;
#pragma omp parallel for if (total > 20000)
        for (idx = 0; idx < (ptrdiff_t)total; idx++)
        {
            data[idx] = val;
        }
#endif
        return;
    }

    int i;
#pragma omp parallel for if (total > 20000)
    for (i = 0; i < r; i++)
    {
        double *row_ptr = &mat->data[(size_t)i * (size_t)mat->stride];
        int j = 0;
#ifdef __AVX2__
        __m256d v = _mm256_set1_pd(val);
        for (; j + 4 <= c; j += 4)
        {
            _mm256_storeu_pd(row_ptr + j, v);
        }
#endif
        for (; j < c; j++)
        {
            row_ptr[j] = val;
        }
    }
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success and a -1 value upon failure.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2)
{
    if (!result || !mat1 || !mat2 ||
        mat1->rows != mat2->rows || mat1->cols != mat2->cols ||
        result->rows != mat1->rows || result->cols != mat1->cols)
    {
        return -1;
    }

    int row = mat1->rows;
    int col = mat2->cols;
    size_t total = (size_t)row * (size_t)col;

    if (result->stride == col && mat1->stride == col && mat2->stride == col)
    {
        double *dst = result->data;
        const double *a = mat1->data;
        const double *b = mat2->data;
#ifdef __AVX2__
        size_t vec_end = total & ~(size_t)3;
        ptrdiff_t idx;
#pragma omp parallel for if (total > 20000)
        for (idx = 0; idx < (ptrdiff_t)vec_end; idx += 4)
        {
            __m256d v1 = _mm256_loadu_pd(a + idx);
            __m256d v2 = _mm256_loadu_pd(b + idx);
            _mm256_storeu_pd(dst + idx, _mm256_add_pd(v1, v2));
        }
        for (size_t j = vec_end; j < total; j++)
        {
            dst[j] = a[j] + b[j];
        }
#else
        ptrdiff_t idx;
#pragma omp parallel for if (total > 20000)
        for (idx = 0; idx < (ptrdiff_t)total; idx++)
        {
            dst[idx] = a[idx] + b[idx];
        }
#endif
        return 0;
    }

    int i;
#pragma omp parallel for if (total > 20000)
    for (i = 0; i < row; i++)
    {
        double *dst = &result->data[(size_t)i * (size_t)result->stride];
        const double *a = &mat1->data[(size_t)i * (size_t)mat1->stride];
        const double *b = &mat2->data[(size_t)i * (size_t)mat2->stride];
        int j = 0;
#ifdef __AVX2__
        for (; j + 4 <= col; j += 4)
        {
            __m256d v1 = _mm256_loadu_pd(a + j);
            __m256d v2 = _mm256_loadu_pd(b + j);
            _mm256_storeu_pd(dst + j, _mm256_add_pd(v1, v2));
        }
#endif
        for (; j < col; j++)
        {
            dst[j] = a[j] + b[j];
        }
    }

    return 0;
}

/*
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success and a -1 value upon failure.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2)
{
    if (!result || !mat1 || !mat2 ||
        mat1->rows != mat2->rows || mat1->cols != mat2->cols ||
        result->rows != mat1->rows || result->cols != mat1->cols)
    {
        return -1;
    }

    int row = mat1->rows;
    int col = mat2->cols;
    size_t total = (size_t)row * (size_t)col;

    if (result->stride == col && mat1->stride == col && mat2->stride == col)
    {
        double *dst = result->data;
        const double *a = mat1->data;
        const double *b = mat2->data;
#ifdef __AVX2__
        size_t vec_end = total & ~(size_t)3;
        ptrdiff_t idx;
#pragma omp parallel for if (total > 20000)
        for (idx = 0; idx < (ptrdiff_t)vec_end; idx += 4)
        {
            __m256d v1 = _mm256_loadu_pd(a + idx);
            __m256d v2 = _mm256_loadu_pd(b + idx);
            _mm256_storeu_pd(dst + idx, _mm256_sub_pd(v1, v2));
        }
        for (size_t j = vec_end; j < total; j++)
        {
            dst[j] = a[j] - b[j];
        }
#else
        ptrdiff_t idx;
#pragma omp parallel for if (total > 20000)
        for (idx = 0; idx < (ptrdiff_t)total; idx++)
        {
            dst[idx] = a[idx] - b[idx];
        }
#endif
        return 0;
    }

    int i;
#pragma omp parallel for if (total > 20000)
    for (i = 0; i < row; i++)
    {
        double *dst = &result->data[(size_t)i * (size_t)result->stride];
        const double *a = &mat1->data[(size_t)i * (size_t)mat1->stride];
        const double *b = &mat2->data[(size_t)i * (size_t)mat2->stride];
        int j = 0;
#ifdef __AVX2__
        for (; j + 4 <= col; j += 4)
        {
            __m256d v1 = _mm256_loadu_pd(a + j);
            __m256d v2 = _mm256_loadu_pd(b + j);
            _mm256_storeu_pd(dst + j, _mm256_sub_pd(v1, v2));
        }
#endif
        for (; j < col; j++)
        {
            dst[j] = a[j] - b[j];
        }
    }

    return 0;
}

#define NUMOX_MC 64
#define NUMOX_NC 128
#define NUMOX_KC 256

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success and a -1 value upon failure.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2)
{
    if (!result || !mat1 || !mat2)
    {
        return -1;
    }
    int row1 = mat1->rows;
    int row2 = mat2->rows;
    int col1 = mat1->cols;
    int col2 = mat2->cols;

    if (col1 != row2 || result->rows != row1 || result->cols != col2)
    {
        return -1;
    }

    int num_bi = (row1 + NUMOX_MC - 1) / NUMOX_MC;
    int num_bj = (col2 + NUMOX_NC - 1) / NUMOX_NC;
    int num_tiles = num_bi * num_bj;
    size_t work = (size_t)row1 * (size_t)col1 * (size_t)col2;

    int tile;
#pragma omp parallel for schedule(dynamic) if (work > 2000000ULL)
    for (tile = 0; tile < num_tiles; tile++)
    {
        int bi = tile / num_bj;
        int bj = tile % num_bj;
        int i0 = bi * NUMOX_MC;
        int j0 = bj * NUMOX_NC;
        int i_end = (i0 + NUMOX_MC < row1) ? (i0 + NUMOX_MC) : row1;
        int j_end = (j0 + NUMOX_NC < col2) ? (j0 + NUMOX_NC) : col2;

        for (int k0 = 0; k0 < col1; k0 += NUMOX_KC)
        {
            int k_end = (k0 + NUMOX_KC < col1) ? (k0 + NUMOX_KC) : col1;
            int k_len = k_end - k0;
            int i = i0;

#ifdef __AVX2__
            for (; i + 4 <= i_end; i += 4)
            {
                const double *a0 = &mat1->data[(size_t)(i + 0) * (size_t)mat1->stride + k0];
                const double *a1 = &mat1->data[(size_t)(i + 1) * (size_t)mat1->stride + k0];
                const double *a2 = &mat1->data[(size_t)(i + 2) * (size_t)mat1->stride + k0];
                const double *a3 = &mat1->data[(size_t)(i + 3) * (size_t)mat1->stride + k0];

                double *c0 = &result->data[(size_t)(i + 0) * (size_t)result->stride];
                double *c1 = &result->data[(size_t)(i + 1) * (size_t)result->stride];
                double *c2 = &result->data[(size_t)(i + 2) * (size_t)result->stride];
                double *c3 = &result->data[(size_t)(i + 3) * (size_t)result->stride];

                int j = j0;
                for (; j + 8 <= j_end; j += 8)
                {
                    __m256d c00, c01, c10, c11, c20, c21, c30, c31;
                    if (k0 == 0)
                    {
                        c00 = _mm256_setzero_pd(); c01 = _mm256_setzero_pd();
                        c10 = _mm256_setzero_pd(); c11 = _mm256_setzero_pd();
                        c20 = _mm256_setzero_pd(); c21 = _mm256_setzero_pd();
                        c30 = _mm256_setzero_pd(); c31 = _mm256_setzero_pd();
                    }
                    else
                    {
                        c00 = _mm256_loadu_pd(c0 + j); c01 = _mm256_loadu_pd(c0 + j + 4);
                        c10 = _mm256_loadu_pd(c1 + j); c11 = _mm256_loadu_pd(c1 + j + 4);
                        c20 = _mm256_loadu_pd(c2 + j); c21 = _mm256_loadu_pd(c2 + j + 4);
                        c30 = _mm256_loadu_pd(c3 + j); c31 = _mm256_loadu_pd(c3 + j + 4);
                    }

                    const double *bk = &mat2->data[(size_t)k0 * (size_t)mat2->stride + j];
                    int b_stride = mat2->stride;
                    for (int k = 0; k < k_len; k++, bk += b_stride)
                    {
                        __m256d b0 = _mm256_loadu_pd(bk);
                        __m256d b1 = _mm256_loadu_pd(bk + 4);
                        __m256d va;
                        va = _mm256_set1_pd(a0[k]);
                        c00 = _mm256_fmadd_pd(va, b0, c00);
                        c01 = _mm256_fmadd_pd(va, b1, c01);
                        va = _mm256_set1_pd(a1[k]);
                        c10 = _mm256_fmadd_pd(va, b0, c10);
                        c11 = _mm256_fmadd_pd(va, b1, c11);
                        va = _mm256_set1_pd(a2[k]);
                        c20 = _mm256_fmadd_pd(va, b0, c20);
                        c21 = _mm256_fmadd_pd(va, b1, c21);
                        va = _mm256_set1_pd(a3[k]);
                        c30 = _mm256_fmadd_pd(va, b0, c30);
                        c31 = _mm256_fmadd_pd(va, b1, c31);
                    }

                    _mm256_storeu_pd(c0 + j, c00); _mm256_storeu_pd(c0 + j + 4, c01);
                    _mm256_storeu_pd(c1 + j, c10); _mm256_storeu_pd(c1 + j + 4, c11);
                    _mm256_storeu_pd(c2 + j, c20); _mm256_storeu_pd(c2 + j + 4, c21);
                    _mm256_storeu_pd(c3 + j, c30); _mm256_storeu_pd(c3 + j + 4, c31);
                }

                for (; j < j_end; j++)
                {
                    double s0 = (k0 == 0) ? 0.0 : c0[j];
                    double s1 = (k0 == 0) ? 0.0 : c1[j];
                    double s2 = (k0 == 0) ? 0.0 : c2[j];
                    double s3 = (k0 == 0) ? 0.0 : c3[j];
                    const double *bk = &mat2->data[(size_t)k0 * (size_t)mat2->stride + j];
                    int b_stride = mat2->stride;
                    for (int k = 0; k < k_len; k++, bk += b_stride)
                    {
                        double bval = *bk;
                        s0 += a0[k] * bval;
                        s1 += a1[k] * bval;
                        s2 += a2[k] * bval;
                        s3 += a3[k] * bval;
                    }
                    c0[j] = s0; c1[j] = s1; c2[j] = s2; c3[j] = s3;
                }
            }
#endif
            for (; i < i_end; i++)
            {
                const double *ai = &mat1->data[(size_t)i * (size_t)mat1->stride + k0];
                double *ci = &result->data[(size_t)i * (size_t)result->stride];
                int j = j0;
#ifdef __AVX2__
                for (; j + 8 <= j_end; j += 8)
                {
                    __m256d c00 = (k0 == 0) ? _mm256_setzero_pd() : _mm256_loadu_pd(ci + j);
                    __m256d c01 = (k0 == 0) ? _mm256_setzero_pd() : _mm256_loadu_pd(ci + j + 4);
                    const double *bk = &mat2->data[(size_t)k0 * (size_t)mat2->stride + j];
                    int b_stride = mat2->stride;
                    for (int k = 0; k < k_len; k++, bk += b_stride)
                    {
                        __m256d va = _mm256_set1_pd(ai[k]);
                        c00 = _mm256_fmadd_pd(va, _mm256_loadu_pd(bk), c00);
                        c01 = _mm256_fmadd_pd(va, _mm256_loadu_pd(bk + 4), c01);
                    }
                    _mm256_storeu_pd(ci + j, c00);
                    _mm256_storeu_pd(ci + j + 4, c01);
                }
#endif
                for (; j < j_end; j++)
                {
                    double sum = (k0 == 0) ? 0.0 : ci[j];
                    const double *bk = &mat2->data[(size_t)k0 * (size_t)mat2->stride + j];
                    int b_stride = mat2->stride;
                    for (int k = 0; k < k_len; k++, bk += b_stride)
                    {
                        sum += ai[k] * (*bk);
                    }
                    ci[j] = sum;
                }
            }
        }
    }

    return 0;
}

static void copy_matrix_data(matrix *dst, const matrix *src)
{
    for (int r = 0; r < src->rows; r++)
    {
        memcpy(&dst->data[(size_t)r * (size_t)dst->stride],
               &src->data[(size_t)r * (size_t)src->stride],
               (size_t)src->cols * sizeof(double));
    }
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success and a -1 value upon failure.
 */
int pow_matrix(matrix *result, matrix *mat, int pow)
{
    if (!result || !mat || mat->rows != mat->cols || pow < 0 ||
        result->rows != mat->rows || result->cols != mat->cols)
    {
        return -1;
    }

    if (pow == 0)
    {
        fill_matrix(result, 0.0);
        for (int i = 0; i < result->rows; i++)
        {
            result->data[(size_t)i * (size_t)result->stride + i] = 1.0;
        }
        return 0;
    }

    if (pow == 1)
    {
        copy_matrix_data(result, mat);
        return 0;
    }

    matrix *base = NULL;
    matrix *base_next = NULL;
    matrix *acc_tmp = NULL;
    if (allocate_matrix_raw(&base, mat->rows, mat->cols) != 0 ||
        allocate_matrix_raw(&base_next, mat->rows, mat->cols) != 0 ||
        allocate_matrix_raw(&acc_tmp, mat->rows, mat->cols) != 0)
    {
        deallocate_matrix(base);
        deallocate_matrix(base_next);
        deallocate_matrix(acc_tmp);
        return -1;
    }

    copy_matrix_data(base, mat);

    fill_matrix(result, 0.0);
    for (int i = 0; i < result->rows; i++)
    {
        result->data[(size_t)i * (size_t)result->stride + i] = 1.0;
    }

    while (pow > 0)
    {
        if (pow & 1)
        {
            mul_matrix(acc_tmp, result, base);
            copy_matrix_data(result, acc_tmp);
        }
        pow >>= 1;
        if (pow > 0)
        {
            mul_matrix(base_next, base, base);
            matrix *swap = base;
            base = base_next;
            base_next = swap;
        }
    }

    deallocate_matrix(base);
    deallocate_matrix(base_next);
    deallocate_matrix(acc_tmp);
    return 0;
}

/*
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success and a -1 value upon failure.
 */
int neg_matrix(matrix *result, matrix *mat)
{
    if (!result || !mat || result->rows != mat->rows || result->cols != mat->cols)
    {
        return -1;
    }
    int row = mat->rows;
    int col = mat->cols;
    size_t total = (size_t)row * (size_t)col;

    if (result->stride == col && mat->stride == col)
    {
        double *dst = result->data;
        const double *src = mat->data;
#ifdef __AVX2__
        __m256d sign_mask = _mm256_set1_pd(-0.0);
        size_t vec_end = total & ~(size_t)3;
        ptrdiff_t idx;
#pragma omp parallel for if (total > 20000)
        for (idx = 0; idx < (ptrdiff_t)vec_end; idx += 4)
        {
            __m256d v = _mm256_loadu_pd(src + idx);
            _mm256_storeu_pd(dst + idx, _mm256_xor_pd(v, sign_mask));
        }
        for (size_t j = vec_end; j < total; j++)
        {
            dst[j] = -src[j];
        }
#else
        ptrdiff_t idx;
#pragma omp parallel for if (total > 20000)
        for (idx = 0; idx < (ptrdiff_t)total; idx++)
        {
            dst[idx] = -src[idx];
        }
#endif
        return 0;
    }

    int i;
#pragma omp parallel for if (total > 20000)
    for (i = 0; i < row; i++)
    {
        double *dst = &result->data[(size_t)i * (size_t)result->stride];
        const double *src = &mat->data[(size_t)i * (size_t)mat->stride];
        int j = 0;
#ifdef __AVX2__
        __m256d sign_mask = _mm256_set1_pd(-0.0);
        for (; j + 4 <= col; j += 4)
        {
            __m256d v = _mm256_loadu_pd(src + j);
            _mm256_storeu_pd(dst + j, _mm256_xor_pd(v, sign_mask));
        }
#endif
        for (; j < col; j++)
        {
            dst[j] = -src[j];
        }
    }

    return 0;
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success and a -1 value upon failure.
 */
int abs_matrix(matrix *result, matrix *mat)
{
    if (!result || !mat || result->rows != mat->rows || result->cols != mat->cols)
    {
        return -1;
    }
    int row = mat->rows;
    int col = mat->cols;
    size_t total = (size_t)row * (size_t)col;

    if (result->stride == col && mat->stride == col)
    {
        double *dst = result->data;
        const double *src = mat->data;
#ifdef __AVX2__
        __m256d sign_mask = _mm256_set1_pd(-0.0);
        size_t vec_end = total & ~(size_t)3;
        ptrdiff_t idx;
#pragma omp parallel for if (total > 20000)
        for (idx = 0; idx < (ptrdiff_t)vec_end; idx += 4)
        {
            __m256d v = _mm256_loadu_pd(src + idx);
            _mm256_storeu_pd(dst + idx, _mm256_andnot_pd(sign_mask, v));
        }
        for (size_t j = vec_end; j < total; j++)
        {
            double num = src[j];
            dst[j] = (num < 0.0) ? -num : num;
        }
#else
        ptrdiff_t idx;
#pragma omp parallel for if (total > 20000)
        for (idx = 0; idx < (ptrdiff_t)total; idx++)
        {
            double num = src[idx];
            dst[idx] = (num < 0.0) ? -num : num;
        }
#endif
        return 0;
    }

    int i;
#pragma omp parallel for if (total > 20000)
    for (i = 0; i < row; i++)
    {
        double *dst = &result->data[(size_t)i * (size_t)result->stride];
        const double *src = &mat->data[(size_t)i * (size_t)mat->stride];
        int j = 0;
#ifdef __AVX2__
        __m256d sign_mask = _mm256_set1_pd(-0.0);
        for (; j + 4 <= col; j += 4)
        {
            __m256d v = _mm256_loadu_pd(src + j);
            _mm256_storeu_pd(dst + j, _mm256_andnot_pd(sign_mask, v));
        }
#endif
        for (; j < col; j++)
        {
            double num = src[j];
            dst[j] = (num < 0.0) ? -num : num;
        }
    }
    return 0;
}
