#include "numox.h"
#include <structmember.h>

PyTypeObject MatrixoxType;

/* Helper functions for initalization of matrices and vectors */

/*
 * Return a tuple given rows and cols
 */
PyObject *get_shape(int rows, int cols)
{
    if (rows == 1 || cols == 1)
    {
        PyObject *dim = PyLong_FromLong((long)rows * (long)cols);
        if (!dim)
            return NULL;
        PyObject *tup = PyTuple_Pack(1, dim);
        Py_DECREF(dim);
        return tup;
    }
    else
    {
        PyObject *r = PyLong_FromLong(rows);
        PyObject *c = PyLong_FromLong(cols);
        if (!r || !c)
        {
            Py_XDECREF(r);
            Py_XDECREF(c);
            return NULL;
        }
        PyObject *tup = PyTuple_Pack(2, r, c);
        Py_DECREF(r);
        Py_DECREF(c);
        return tup;
    }
}

static int py_to_double(PyObject *obj, double *out)
{
    if (PyFloat_Check(obj))
    {
        *out = PyFloat_AS_DOUBLE(obj);
        return 0;
    }
    if (PyLong_Check(obj))
    {
        double val = PyLong_AsDouble(obj);
        if (val == -1.0 && PyErr_Occurred())
        {
            return -1;
        }
        *out = val;
        return 0;
    }
    PyErr_SetString(PyExc_TypeError, "Matrix elements must be int or float");
    return -1;
}
/*
 * Matrix(rows, cols, low, high). Fill a matrix random double values
 */
int init_rand(PyObject *self, int rows, int cols, unsigned int seed, double low,
              double high)
{
    if (rows <= 0 || cols <= 0)
    {
        PyErr_SetString(PyExc_ValueError, "Matrix dimensions must be positive");
        return -1;
    }
    matrix *new_mat = NULL;
    if (allocate_matrix(&new_mat, rows, cols) != 0)
    {
        PyErr_NoMemory();
        return -1;
    }
    Py_BEGIN_ALLOW_THREADS
        rand_matrix(new_mat, seed, low, high);
    Py_END_ALLOW_THREADS
    PyObject *shape = get_shape(new_mat->rows, new_mat->cols);
    if (!shape)
    {
        deallocate_matrix(new_mat);
        return -1;
    }
    ((Matrixox *)self)->mat = new_mat;
    ((Matrixox *)self)->shape = shape;
    return 0;
}

/*
 * Matrix(rows, cols, val). Fill a matrix of dimension rows * cols with val
 */
int init_fill(PyObject *self, int rows, int cols, double val)
{
    if (rows <= 0 || cols <= 0)
    {
        PyErr_SetString(PyExc_ValueError, "Matrix dimensions must be positive");
        return -1;
    }
    matrix *new_mat = NULL;
    if (allocate_matrix_raw(&new_mat, rows, cols) != 0)
    {
        PyErr_NoMemory();
        return -1;
    }
    Py_BEGIN_ALLOW_THREADS
        fill_matrix(new_mat, val);
    Py_END_ALLOW_THREADS
    PyObject *shape = get_shape(new_mat->rows, new_mat->cols);
    if (!shape)
    {
        deallocate_matrix(new_mat);
        return -1;
    }
    ((Matrixox *)self)->mat = new_mat;
    ((Matrixox *)self)->shape = shape;
    return 0;
}

/*
 * Matrix(rows, cols, 1d_list). Fill a matrix with dimension rows * cols with 1d_list values
 */
int init_1d(PyObject *self, int rows, int cols, PyObject *lst)
{
    if (rows <= 0 || cols <= 0)
    {
        PyErr_SetString(PyExc_ValueError, "Matrix dimensions must be positive");
        return -1;
    }
    if ((Py_ssize_t)rows * (Py_ssize_t)cols != PyList_Size(lst))
    {
        PyErr_SetString(PyExc_ValueError, "Incorrect number of elements in list");
        return -1;
    }
    matrix *new_mat = NULL;
    if (allocate_matrix_raw(&new_mat, rows, cols) != 0)
    {
        PyErr_NoMemory();
        return -1;
    }
    Py_ssize_t count = 0;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            double val;
            if (py_to_double(PyList_GET_ITEM(lst, count), &val) != 0)
            {
                deallocate_matrix(new_mat);
                return -1;
            }
            set(new_mat, i, j, val);
            count++;
        }
    }
    PyObject *shape = get_shape(new_mat->rows, new_mat->cols);
    if (!shape)
    {
        deallocate_matrix(new_mat);
        return -1;
    }
    ((Matrixox *)self)->mat = new_mat;
    ((Matrixox *)self)->shape = shape;
    return 0;
}

/*
 * Matrix(2d_list). Fill a matrix with dimension len(2d_list) * len(2d_list[0])
 */
int init_2d(PyObject *self, PyObject *lst)
{
    Py_ssize_t rows_sz = PyList_Size(lst);
    if (rows_sz <= 0)
    {
        PyErr_SetString(PyExc_ValueError,
                        "Cannot initialize numox.Matrix with an empty list");
        return -1;
    }
    PyObject *first_row = PyList_GET_ITEM(lst, 0);
    if (!PyList_Check(first_row))
    {
        PyErr_SetString(PyExc_ValueError, "List values not valid");
        return -1;
    }
    Py_ssize_t cols_sz = PyList_Size(first_row);
    if (cols_sz <= 0)
    {
        PyErr_SetString(PyExc_ValueError, "List values not valid");
        return -1;
    }
    int rows = (int)rows_sz;
    int cols = (int)cols_sz;
    for (int i = 0; i < rows; i++)
    {
        PyObject *row_obj = PyList_GET_ITEM(lst, i);
        if (!PyList_Check(row_obj) || PyList_Size(row_obj) != cols_sz)
        {
            PyErr_SetString(PyExc_ValueError, "List values not valid");
            return -1;
        }
    }
    matrix *new_mat = NULL;
    if (allocate_matrix_raw(&new_mat, rows, cols) != 0)
    {
        PyErr_NoMemory();
        return -1;
    }
    for (int i = 0; i < rows; i++)
    {
        PyObject *row_obj = PyList_GET_ITEM(lst, i);
        for (int j = 0; j < cols; j++)
        {
            double val;
            if (py_to_double(PyList_GET_ITEM(row_obj, j), &val) != 0)
            {
                deallocate_matrix(new_mat);
                return -1;
            }
            set(new_mat, i, j, val);
        }
    }
    PyObject *shape = get_shape(new_mat->rows, new_mat->cols);
    if (!shape)
    {
        deallocate_matrix(new_mat);
        return -1;
    }
    ((Matrixox *)self)->mat = new_mat;
    ((Matrixox *)self)->shape = shape;
    return 0;
}

/*
 * This deallocation function is called when reference count is 0
 */
void Matrixox_dealloc(Matrixox *self)
{
    deallocate_matrix(self->mat);
    Py_XDECREF(self->shape);
    Py_TYPE(self)->tp_free(self);
}

/* For immutable types all initializations should take place in tp_new */
PyObject *Matrixox_new(PyTypeObject *type, PyObject *args,
                       PyObject *kwds)
{
    /* size of allocated memory is tp_basicsize + nitems*tp_itemsize*/
    Matrixox *self = (Matrixox *)type->tp_alloc(type, 0);
    return (PyObject *)self;
}

/*
 * This matrixox type is mutable, so needs init function. Return 0 on success otherwise -1
 */
int Matrixox_init(PyObject *self, PyObject *args, PyObject *kwds)
{
    Matrixox *m = (Matrixox *)self;
    if (m->mat != NULL)
    {
        deallocate_matrix(m->mat);
        m->mat = NULL;
    }
    Py_CLEAR(m->shape);

    /* Generate random matrices */
    if (kwds != NULL && PyDict_Size(kwds) > 0)
    {
        PyObject *rand = PyDict_GetItemString(kwds, "rand");
        if (!rand || !PyBool_Check(rand) || rand != Py_True)
        {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }

        PyObject *low = PyDict_GetItemString(kwds, "low");
        PyObject *high = PyDict_GetItemString(kwds, "high");
        PyObject *seed = PyDict_GetItemString(kwds, "seed");
        double double_low = 0.0;
        double double_high = 1.0;
        unsigned int unsigned_seed = 0;

        if (low && py_to_double(low, &double_low) != 0)
        {
            return -1;
        }
        if (high && py_to_double(high, &double_high) != 0)
        {
            return -1;
        }

        if (double_low >= double_high)
        {
            PyErr_SetString(PyExc_ValueError, "Invalid arguments: low must be less than high");
            return -1;
        }

        if (seed)
        {
            if (!PyLong_Check(seed))
            {
                PyErr_SetString(PyExc_TypeError, "seed must be an integer");
                return -1;
            }
            unsigned_seed = (unsigned int)PyLong_AsUnsignedLongMask(seed);
        }

        PyObject *rows = NULL;
        PyObject *cols = NULL;
        if (!PyArg_UnpackTuple(args, "Matrix", 2, 2, &rows, &cols))
        {
            return -1;
        }
        if (PyLong_Check(rows) && PyLong_Check(cols))
        {
            return init_rand(self, (int)PyLong_AsLong(rows), (int)PyLong_AsLong(cols),
                             unsigned_seed, double_low, double_high);
        }
        PyErr_SetString(PyExc_TypeError, "Invalid arguments");
        return -1;
    }
    PyObject *arg1 = NULL;
    PyObject *arg2 = NULL;
    PyObject *arg3 = NULL;
    if (PyArg_UnpackTuple(args, "args", 1, 3, &arg1, &arg2, &arg3))
    {
        /* arguments are (rows, cols, val) */
        if (arg1 && arg2 && arg3 && PyLong_Check(arg1) && PyLong_Check(arg2) && (PyLong_Check(arg3) || PyFloat_Check(arg3)))
        {
            if (PyLong_Check(arg3))
            {
                return init_fill(self, PyLong_AsLong(arg1), PyLong_AsLong(arg2), PyLong_AsLong(arg3));
            }
            else
                return init_fill(self, PyLong_AsLong(arg1), PyLong_AsLong(arg2), PyFloat_AsDouble(arg3));
        }
        else if (arg1 && arg2 && arg3 && PyLong_Check(arg1) && PyLong_Check(arg2) && PyList_Check(arg3))
        {
            /* Matrix(rows, cols, 1D list) */
            return init_1d(self, PyLong_AsLong(arg1), PyLong_AsLong(arg2), arg3);
        }
        else if (arg1 && PyList_Check(arg1) && arg2 == NULL && arg3 == NULL)
        {
            /* Matrix(rows, cols, 1D list) */
            return init_2d(self, arg1);
        }
        else if (arg1 && arg2 && PyLong_Check(arg1) && PyLong_Check(arg2) && arg3 == NULL)
        {
            /* Matrix(rows, cols, 1D list) */
            return init_fill(self, PyLong_AsLong(arg1), PyLong_AsLong(arg2), 0);
        }
        else
        {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Invalid arguments");
        return -1;
    }
}

/*
 * List of lists representations for matrices
 */
PyObject *Matrixox_to_list(Matrixox *self)
{
    int rows = self->mat->rows;
    int cols = self->mat->cols;
    PyObject *py_lst = NULL;
    if (self->mat->is_1d)
    { // If 1D matrix, print as a single list
        py_lst = PyList_New(rows * cols);
        int count = 0;
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                PyList_SetItem(py_lst, count, PyFloat_FromDouble(get(self->mat, i, j)));
                count++;
            }
        }
    }
    else
    { // if 2D, print as nested list
        py_lst = PyList_New(rows);
        for (int i = 0; i < rows; i++)
        {
            PyList_SetItem(py_lst, i, PyList_New(cols));
            PyObject *curr_row = PyList_GetItem(py_lst, i);
            for (int j = 0; j < cols; j++)
            {
                PyList_SetItem(curr_row, j, PyFloat_FromDouble(get(self->mat, i, j)));
            }
        }
    }
    return py_lst;
}

PyObject *Matrixox_class_to_list(Matrixox *self, PyObject *args)
{
    PyObject *mat = NULL;
    if (PyArg_UnpackTuple(args, "args", 1, 1, &mat))
    {
        if (!PyObject_TypeCheck(mat, &MatrixoxType))
        {
            PyErr_SetString(PyExc_TypeError, "Argument must of type numox.Matrix!");
            return NULL;
        }
        Matrixox *matox = (Matrixox *)mat;
        return Matrixox_to_list(matox);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Invalid arguments");
        return NULL;
    }
}

/*
 * Add class methods
 */
PyMethodDef Matrixox_class_methods[] = {
    {"to_list", (PyCFunction)Matrixox_class_to_list, METH_VARARGS, "Returns a list representation of numox.Matrix"},
    {NULL, NULL, 0, NULL}};

/*
 * Matrixox string representation. For printing purposes.
 */
PyObject *Matrixox_repr(PyObject *self)
{
    PyObject *py_lst = Matrixox_to_list((Matrixox *)self);
    if (!py_lst)
    {
        return NULL;
    }
    PyObject *res = PyObject_Repr(py_lst);
    Py_DECREF(py_lst);
    return res;
}

/* Helper to allocate a Matrixox object and its underlying raw matrix + shape */
static Matrixox *create_matrixox(int rows, int cols)
{
    Matrixox *ans = (Matrixox *)MatrixoxType.tp_alloc(&MatrixoxType, 0);
    if (!ans)
    {
        return NULL;
    }
    ans->mat = NULL;
    ans->shape = NULL;
    if (allocate_matrix_raw(&ans->mat, rows, cols) != 0)
    {
        Py_DECREF(ans);
        PyErr_NoMemory();
        return NULL;
    }
    ans->shape = get_shape(rows, cols);
    if (!ans->shape)
    {
        Py_DECREF(ans);
        return NULL;
    }
    return ans;
}

/* NUMBER METHODS */

/*
 * Add the second numox.Matrix (Matrixox) object to the first one. The first operand is
 * self, and the second operand can be obtained by casting `args`.
 */
PyObject *Matrixox_add(Matrixox *self, PyObject *args)
{
    if (!PyObject_TypeCheck(args, &MatrixoxType))
    {
        PyErr_SetString(PyExc_TypeError, "Operand must be a numox.Matrix");
        return NULL;
    }

    Matrixox *other = (Matrixox *)args;

    if (self->mat->rows != other->mat->rows ||
        self->mat->cols != other->mat->cols)
    {
        PyErr_SetString(PyExc_ValueError, "Matrix dimensions do not match");
        return NULL;
    }

    Matrixox *ans = create_matrixox(self->mat->rows, self->mat->cols);
    if (!ans)
    {
        return NULL;
    }

    int err;
    Py_BEGIN_ALLOW_THREADS
        err = add_matrix(ans->mat, self->mat, other->mat);
    Py_END_ALLOW_THREADS

    if (err != 0)
    {
        Py_DECREF(ans);
        PyErr_SetString(PyExc_RuntimeError, "Matrix addition failed");
        return NULL;
    }

    return (PyObject *)ans;
}

/*
 * Substract the second numox.Matrix (Matrixox) object from the first one. The first operand is
 * self, and the second operand can be obtained by casting `args`.
 */
PyObject *Matrixox_sub(Matrixox *self, PyObject *args)
{
    if (!PyObject_TypeCheck(args, &MatrixoxType))
    {
        PyErr_SetString(PyExc_TypeError, "Operand must be a numox.Matrix");
        return NULL;
    }

    Matrixox *other = (Matrixox *)args;

    if (self->mat->rows != other->mat->rows ||
        self->mat->cols != other->mat->cols)
    {
        PyErr_SetString(PyExc_ValueError, "Matrix dimensions do not match");
        return NULL;
    }

    Matrixox *ans = create_matrixox(self->mat->rows, self->mat->cols);
    if (!ans)
    {
        return NULL;
    }

    int err;
    Py_BEGIN_ALLOW_THREADS
        err = sub_matrix(ans->mat, self->mat, other->mat);
    Py_END_ALLOW_THREADS

    if (err != 0)
    {
        Py_DECREF(ans);
        PyErr_SetString(PyExc_RuntimeError, "Matrix subtraction failed");
        return NULL;
    }

    return (PyObject *)ans;
}

/*
 * NOT element-wise multiplication. The first operand is self, and the second operand
 * can be obtained by casting `args`.
 */
PyObject *Matrixox_multiply(Matrixox *self, PyObject *args)
{
    if (!PyObject_TypeCheck(args, &MatrixoxType))
    {
        PyErr_SetString(PyExc_TypeError, "Operand must be a numox.Matrix");
        return NULL;
    }

    Matrixox *other = (Matrixox *)args;

    if (self->mat->cols != other->mat->rows)
    {
        PyErr_SetString(PyExc_ValueError, "Matrix dimensions do not match");
        return NULL;
    }

    Matrixox *ans = create_matrixox(self->mat->rows, other->mat->cols);
    if (!ans)
    {
        return NULL;
    }

    int err;
    Py_BEGIN_ALLOW_THREADS
        err = mul_matrix(ans->mat, self->mat, other->mat);
    Py_END_ALLOW_THREADS

    if (err != 0)
    {
        Py_DECREF(ans);
        PyErr_NoMemory();
        return NULL;
    }

    return (PyObject *)ans;
}

/*
 * Negates the given numox.Matrix.
 */
PyObject *Matrixox_neg(Matrixox *self)
{
    Matrixox *ans = create_matrixox(self->mat->rows, self->mat->cols);
    if (!ans)
    {
        return NULL;
    }

    int err;
    Py_BEGIN_ALLOW_THREADS
        err = neg_matrix(ans->mat, self->mat);
    Py_END_ALLOW_THREADS

    if (err != 0)
    {
        Py_DECREF(ans);
        PyErr_SetString(PyExc_RuntimeError, "Matrix negation failed");
        return NULL;
    }

    return (PyObject *)ans;
}

/*
 * Take the element-wise absolute value of this numox.Matrix.
 */
PyObject *Matrixox_abs(Matrixox *self)
{
    Matrixox *ans = create_matrixox(self->mat->rows, self->mat->cols);
    if (!ans)
    {
        return NULL;
    }

    int err;
    Py_BEGIN_ALLOW_THREADS
        err = abs_matrix(ans->mat, self->mat);
    Py_END_ALLOW_THREADS

    if (err != 0)
    {
        Py_DECREF(ans);
        PyErr_SetString(PyExc_RuntimeError, "Matrix abs failed");
        return NULL;
    }

    return (PyObject *)ans;
}

/*
 * Raise numox.Matrix (Matrixox) to the `pow`th power. You can ignore the argument `optional`.
 */
PyObject *Matrixox_pow(Matrixox *self, PyObject *pow, PyObject *optional)
{
    if (optional != Py_None)
    {
        PyErr_SetString(PyExc_TypeError, "3-argument pow() is not supported");
        return NULL;
    }
    if (!PyLong_Check(pow))
    {
        PyErr_SetString(PyExc_TypeError, "Exponent must be an integer");
        return NULL;
    }
    long p = PyLong_AsLong(pow);
    if (p == -1 && PyErr_Occurred())
    {
        return NULL;
    }
    if (self->mat->rows != self->mat->cols || p < 0 || p > INT_MAX)
    {
        PyErr_SetString(PyExc_ValueError, "Matrix must be square and exponent must be a non-negative int");
        return NULL;
    }

    Matrixox *ans = create_matrixox(self->mat->rows, self->mat->cols);
    if (!ans)
    {
        return NULL;
    }

    int err;
    Py_BEGIN_ALLOW_THREADS
        err = pow_matrix(ans->mat, self->mat, (int)p);
    Py_END_ALLOW_THREADS

    if (err != 0)
    {
        Py_DECREF(ans);
        PyErr_NoMemory();
        return NULL;
    }

    return (PyObject *)ans;
}

/*
 * Create a PyNumberMethods struct for overloading operators with all the number methods you have
 * define. You might find this link helpful: https://docs.python.org/3.6/c-api/typeobj.html
 */
PyNumberMethods Matrixox_as_number = {
    .nb_add = (binaryfunc)Matrixox_add,
    .nb_subtract = (binaryfunc)Matrixox_sub,
    .nb_multiply = (binaryfunc)Matrixox_multiply,
    .nb_absolute = (unaryfunc)Matrixox_abs,
    .nb_negative = (unaryfunc)Matrixox_neg,
    .nb_power = (ternaryfunc)Matrixox_pow,
    .nb_matrix_multiply = (binaryfunc)Matrixox_multiply,
};

/* INSTANCE METHODS */

/*
 * Given a numox.Matrix self, parse `args` to (int) row, (int) col, and (double/int) val.
 * Return None in Python (this is different from returning null).
 */
PyObject *Matrixox_set_value(Matrixox *self, PyObject *args)
{
    PyObject *r = NULL;
    PyObject *c = NULL;
    PyObject *v = NULL;

    if (!PyArg_UnpackTuple(args, "set", 3, 3, &r, &c, &v))
    {
        return NULL;
    }

    if (!PyLong_Check(r) || !PyLong_Check(c) ||
        (!PyLong_Check(v) && !PyFloat_Check(v)))
    {
        PyErr_SetString(PyExc_TypeError, "Invalid argument types");
        return NULL;
    }

    int row = PyLong_AsLong(r);
    int col = PyLong_AS_LONG(c);
    double val = PyLong_Check(v) ? PyLong_AS_LONG(v) : PyFloat_AS_DOUBLE(v);

    if (row < 0 || col < 0 || row >= self->mat->rows || col >= self->mat->cols)
    {
        PyErr_SetString(PyExc_IndexError, "row or column out of range");
        return NULL;
    }

    set(self->mat, row, col, val);

    Py_RETURN_NONE;
}

/*
 * Given a numox.Matrix `self`, parse `args` to (int) row and (int) col.
 * Return the value at the `row`th row and `col`th column, which is a Python
 * float/int.
 */
PyObject *Matrixox_get_value(Matrixox *self, PyObject *args)
{
    PyObject *r = NULL;
    PyObject *c = NULL;

    if (!PyArg_UnpackTuple(args, "get", 2, 2, &r, &c))
    {
        return NULL;
    }

    if (!PyLong_Check(r) || !PyLong_Check(c))
    {
        PyErr_SetString(PyExc_TypeError, "Invalid argument types");
        return NULL;
    }

    int row = PyLong_AsLong(r);
    int col = PyLong_AS_LONG(c);

    if (row < 0 || col < 0 || row >= self->mat->rows || col >= self->mat->cols)
    {
        PyErr_SetString(PyExc_IndexError, "row or column out of range");
        return NULL;
    }

    double ans = get(self->mat, row, col);
    PyObject *res = PyFloat_FromDouble(ans);

    return res;
}

/*
 * Create an array of PyMethodDef structs to hold the instance methods.
 * Name the python function corresponding to Matrixox_get_value as "get" and Matrixox_set_value
 * as "set"
 * You might find this link helpful: https://docs.python.org/3.6/c-api/structures.html
 */
PyMethodDef Matrixox_methods[] = {
    {"get", (PyCFunction)Matrixox_get_value, METH_VARARGS, "Gets a value from the matrix at (row, col)"},
    {"set", (PyCFunction)Matrixox_set_value, METH_VARARGS, "Sets a value in the matrix at (row, col, val)"},
    {"to_list", (PyCFunction)Matrixox_to_list, METH_NOARGS, "Return a list representation of numox.Matrix"},
    {NULL, NULL, 0, NULL}};

/* INDEXING HELPERS */

static int parse_int_index(PyObject *key, int len, int *out_idx)
{
    if (!PyLong_Check(key))
    {
        PyErr_SetString(PyExc_TypeError, "Index must be an integer");
        return -1;
    }
    long idx = PyLong_AsLong(key);
    if (idx == -1 && PyErr_Occurred())
    {
        return -1;
    }
    if (idx < 0 || idx >= len)
    {
        PyErr_SetString(PyExc_IndexError, "Index out of range");
        return -1;
    }
    *out_idx = (int)idx;
    return 0;
}

static int parse_slice_step1(PyObject *slice_obj, Py_ssize_t len, Py_ssize_t *start, Py_ssize_t *slicelength)
{
    Py_ssize_t stop = 0, step = 0;
    if (PySlice_GetIndicesEx(slice_obj, len, start, &stop, &step, slicelength) < 0)
    {
        return -1;
    }
    if (step != 1)
    {
        PyErr_SetString(PyExc_ValueError, "Slice step must be 1");
        return -1;
    }
    if (*slicelength < 1)
    {
        PyErr_SetString(PyExc_ValueError, "Slice cannot result in empty dimension");
        return -1;
    }
    return 0;
}

static PyObject *create_matrixox_ref(matrix *from, int row_offset, int col_offset, int rows, int cols)
{
    Matrixox *result = (Matrixox *)MatrixoxType.tp_alloc(&MatrixoxType, 0);
    if (!result)
    {
        return NULL;
    }
    result->mat = NULL;
    result->shape = NULL;

    if (allocate_matrix_ref(&(result->mat), from, row_offset, col_offset, rows, cols) != 0)
    {
        Py_DECREF(result);
        PyErr_SetString(PyExc_ValueError, "Failed to create matrix slice");
        return NULL;
    }

    result->shape = get_shape(result->mat->rows, result->mat->cols);
    if (!result->shape)
    {
        Py_DECREF(result);
        return NULL;
    }
    return (PyObject *)result;
}

/*
 * Given a numox.Matrix `self`, index into it with `key`. Return the indexed result.
 */
PyObject *Matrixox_subscript(Matrixox *self, PyObject *key)
{
    if (self->mat->is_1d == 1)
    {
        int len = (self->mat->rows == 1) ? self->mat->cols : self->mat->rows;

        if (PyLong_Check(key))
        {
            int num = 0;
            if (parse_int_index(key, len, &num) != 0)
            {
                return NULL;
            }
            double val = (self->mat->rows == 1) ? get(self->mat, 0, num) : get(self->mat, num, 0);
            return PyFloat_FromDouble(val);
        }

        if (PySlice_Check(key))
        {
            Py_ssize_t start = 0, slicelength = 0;
            if (parse_slice_step1(key, len, &start, &slicelength) != 0)
            {
                return NULL;
            }
            int row_offset = (self->mat->rows == 1) ? 0 : (int)start;
            int col_offset = (self->mat->rows == 1) ? (int)start : 0;
            int new_rows = (self->mat->rows == 1) ? 1 : (int)slicelength;
            int new_cols = (self->mat->rows == 1) ? (int)slicelength : 1;
            return create_matrixox_ref(self->mat, row_offset, col_offset, new_rows, new_cols);
        }

        PyErr_SetString(PyExc_TypeError, "Invalid index type");
        return NULL;
    }
    else
    {
        int row_len = self->mat->rows;
        int col_len = self->mat->cols;

        if (PyLong_Check(key))
        {
            int num = 0;
            if (parse_int_index(key, row_len, &num) != 0)
            {
                return NULL;
            }
            return create_matrixox_ref(self->mat, num, 0, 1, col_len);
        }
        if (PySlice_Check(key))
        {
            Py_ssize_t start = 0, slicelength = 0;
            if (parse_slice_step1(key, row_len, &start, &slicelength) != 0)
            {
                return NULL;
            }
            return create_matrixox_ref(self->mat, (int)start, 0, (int)slicelength, col_len);
        }

        if (PyTuple_Check(key))
        {
            if (PyTuple_Size(key) != 2)
            {
                PyErr_SetString(PyExc_TypeError, "2D matrix require 2 slice/index arguments");
                return NULL;
            }

            PyObject *row_key = PyTuple_GET_ITEM(key, 0);
            PyObject *col_key = PyTuple_GET_ITEM(key, 1);

            if (PyLong_Check(row_key) && PyLong_Check(col_key))
            {
                int r = 0, c = 0;
                if (parse_int_index(row_key, row_len, &r) != 0 ||
                    parse_int_index(col_key, col_len, &c) != 0)
                {
                    return NULL;
                }
                return PyFloat_FromDouble(get(self->mat, r, c));
            }
            if (PyLong_Check(row_key) && PySlice_Check(col_key))
            {
                int r = 0;
                Py_ssize_t start = 0, slicelength = 0;
                if (parse_int_index(row_key, row_len, &r) != 0 ||
                    parse_slice_step1(col_key, col_len, &start, &slicelength) != 0)
                {
                    return NULL;
                }
                return create_matrixox_ref(self->mat, r, (int)start, 1, (int)slicelength);
            }
            if (PySlice_Check(row_key) && PyLong_Check(col_key))
            {
                int c = 0;
                Py_ssize_t start = 0, slicelength = 0;
                if (parse_slice_step1(row_key, row_len, &start, &slicelength) != 0 ||
                    parse_int_index(col_key, col_len, &c) != 0)
                {
                    return NULL;
                }
                return create_matrixox_ref(self->mat, (int)start, c, (int)slicelength, 1);
            }
            if (PySlice_Check(row_key) && PySlice_Check(col_key))
            {
                Py_ssize_t start1 = 0, slicelength1 = 0;
                Py_ssize_t start2 = 0, slicelength2 = 0;
                if (parse_slice_step1(row_key, row_len, &start1, &slicelength1) != 0 ||
                    parse_slice_step1(col_key, col_len, &start2, &slicelength2) != 0)
                {
                    return NULL;
                }
                return create_matrixox_ref(self->mat, (int)start1, (int)start2, (int)slicelength1, (int)slicelength2);
            }

            PyErr_SetString(PyExc_TypeError, "Invalid index type");
            return NULL;
        }
        PyErr_SetString(PyExc_TypeError, "Invalid key type");
        return NULL;
    }
}

/* Helper to validate and write a 1D Python list into a matrix slice */
static int set_1d_list_values(matrix *mat, int row_start, int col_start, int is_row_vec, PyObject *v, Py_ssize_t expected_len)
{
    if (!PyList_Check(v))
    {
        PyErr_SetString(PyExc_TypeError, "Value must be a list");
        return -1;
    }
    if (PyList_Size(v) != expected_len)
    {
        PyErr_SetString(PyExc_ValueError, "Size of value list does not match slice length");
        return -1;
    }
    /* Validate all elements before mutating */
    for (Py_ssize_t i = 0; i < expected_len; i++)
    {
        double tmp;
        if (py_to_double(PyList_GET_ITEM(v, i), &tmp) != 0)
        {
            return -1;
        }
    }
    for (Py_ssize_t i = 0; i < expected_len; i++)
    {
        double val;
        py_to_double(PyList_GET_ITEM(v, i), &val);
        if (is_row_vec)
        {
            set(mat, row_start, col_start + (int)i, val);
        }
        else
        {
            set(mat, row_start + (int)i, col_start, val);
        }
    }
    return 0;
}

/* Helper to validate and write a 2D list/tuple of lists into a matrix slice */
static int set_2d_seq_values(matrix *mat, int row_start, int col_start, Py_ssize_t expected_rows, Py_ssize_t expected_cols, PyObject *v)
{
    if (!PyList_Check(v) && !PyTuple_Check(v))
    {
        PyErr_SetString(PyExc_TypeError, "Value must be a list or tuple of lists");
        return -1;
    }
    Py_ssize_t rn = PySequence_Size(v);
    if (rn != expected_rows)
    {
        PyErr_SetString(PyExc_ValueError, "Row count of value does not match slice");
        return -1;
    }
    for (Py_ssize_t i = 0; i < rn; i++)
    {
        PyObject *row_obj = PyList_Check(v) ? PyList_GET_ITEM(v, i) : PyTuple_GET_ITEM(v, i);
        if (!PyList_Check(row_obj) && !PyTuple_Check(row_obj))
        {
            PyErr_SetString(PyExc_TypeError, "Each row in value must be a list or tuple");
            return -1;
        }
        Py_ssize_t cn = PySequence_Size(row_obj);
        if (cn != expected_cols)
        {
            PyErr_SetString(PyExc_ValueError, "Column count of value does not match slice");
            return -1;
        }
        for (Py_ssize_t j = 0; j < cn; j++)
        {
            PyObject *elem = PyList_Check(row_obj) ? PyList_GET_ITEM(row_obj, j) : PyTuple_GET_ITEM(row_obj, j);
            double tmp;
            if (py_to_double(elem, &tmp) != 0)
            {
                return -1;
            }
        }
    }
    for (Py_ssize_t i = 0; i < rn; i++)
    {
        PyObject *row_obj = PyList_Check(v) ? PyList_GET_ITEM(v, i) : PyTuple_GET_ITEM(v, i);
        for (Py_ssize_t j = 0; j < expected_cols; j++)
        {
            PyObject *elem = PyList_Check(row_obj) ? PyList_GET_ITEM(row_obj, j) : PyTuple_GET_ITEM(row_obj, j);
            double val;
            py_to_double(elem, &val);
            set(mat, row_start + (int)i, col_start + (int)j, val);
        }
    }
    return 0;
}

/*
 * Given a numox.Matrix `self`, index into it with `key`, and set the indexed result to `v`.
 */
int Matrixox_set_subscript(Matrixox *self, PyObject *key, PyObject *v)
{
    if (!v)
    {
        PyErr_SetString(PyExc_TypeError, "Deleting matrix elements is not supported");
        return -1;
    }

    if (self->mat->is_1d)
    {
        int len = (self->mat->rows == 1) ? self->mat->cols : self->mat->rows;
        int is_row_vec = (self->mat->rows == 1);

        if (PyLong_Check(key))
        {
            int idx = 0;
            double val = 0.0;
            if (parse_int_index(key, len, &idx) != 0 || py_to_double(v, &val) != 0)
            {
                return -1;
            }
            if (is_row_vec)
            {
                set(self->mat, 0, idx, val);
            }
            else
            {
                set(self->mat, idx, 0, val);
            }
            return 0;
        }
        if (PySlice_Check(key))
        {
            Py_ssize_t start = 0, slicelength = 0;
            if (parse_slice_step1(key, len, &start, &slicelength) != 0)
            {
                return -1;
            }
            int row_start = is_row_vec ? 0 : (int)start;
            int col_start = is_row_vec ? (int)start : 0;
            return set_1d_list_values(self->mat, row_start, col_start, is_row_vec, v, slicelength);
        }

        PyErr_SetString(PyExc_TypeError, "Invalid key type");
        return -1;
    }
    else
    {
        int row_len = self->mat->rows;
        int col_len = self->mat->cols;

        if (PyLong_Check(key))
        {
            int r = 0;
            if (parse_int_index(key, row_len, &r) != 0)
            {
                return -1;
            }
            return set_1d_list_values(self->mat, r, 0, 1, v, col_len);
        }
        if (PySlice_Check(key))
        {
            Py_ssize_t start = 0, slicelength = 0;
            if (parse_slice_step1(key, row_len, &start, &slicelength) != 0)
            {
                return -1;
            }
            return set_2d_seq_values(self->mat, (int)start, 0, slicelength, col_len, v);
        }
        if (PyTuple_Check(key))
        {
            if (PyTuple_Size(key) != 2)
            {
                PyErr_SetString(PyExc_ValueError, "2D matrix requires 2 slice/index arguments");
                return -1;
            }
            PyObject *key1 = PyTuple_GET_ITEM(key, 0);
            PyObject *key2 = PyTuple_GET_ITEM(key, 1);

            if (PyLong_Check(key1) && PyLong_Check(key2))
            {
                int r = 0, c = 0;
                double val = 0.0;
                if (parse_int_index(key1, row_len, &r) != 0 ||
                    parse_int_index(key2, col_len, &c) != 0 ||
                    py_to_double(v, &val) != 0)
                {
                    return -1;
                }
                set(self->mat, r, c, val);
                return 0;
            }
            if (PyLong_Check(key1) && PySlice_Check(key2))
            {
                int r = 0;
                Py_ssize_t start = 0, slicelength = 0;
                if (parse_int_index(key1, row_len, &r) != 0 ||
                    parse_slice_step1(key2, col_len, &start, &slicelength) != 0)
                {
                    return -1;
                }
                return set_1d_list_values(self->mat, r, (int)start, 1, v, slicelength);
            }
            if (PySlice_Check(key1) && PyLong_Check(key2))
            {
                int c = 0;
                Py_ssize_t start = 0, slicelength = 0;
                if (parse_slice_step1(key1, row_len, &start, &slicelength) != 0 ||
                    parse_int_index(key2, col_len, &c) != 0)
                {
                    return -1;
                }
                return set_1d_list_values(self->mat, (int)start, c, 0, v, slicelength);
            }
            if (PySlice_Check(key1) && PySlice_Check(key2))
            {
                Py_ssize_t start1 = 0, slicelength1 = 0;
                Py_ssize_t start2 = 0, slicelength2 = 0;
                if (parse_slice_step1(key1, row_len, &start1, &slicelength1) != 0 ||
                    parse_slice_step1(key2, col_len, &start2, &slicelength2) != 0)
                {
                    return -1;
                }
                return set_2d_seq_values(self->mat, (int)start1, (int)start2, slicelength1, slicelength2, v);
            }

            PyErr_SetString(PyExc_TypeError, "Invalid index type");
            return -1;
        }

        PyErr_SetString(PyExc_TypeError, "The type of key is not suitable");
        return -1;
    }
}

PyMappingMethods Matrixox_mapping = {
    NULL,
    (binaryfunc)Matrixox_subscript,
    (objobjargproc)Matrixox_set_subscript,
};

/* BUFFER PROTOCOL */
typedef struct {
    Py_ssize_t shape[2];
    Py_ssize_t strides[2];
} MatrixoxBufferInfo;

static int Matrixox_getbuffer(Matrixox *self, Py_buffer *view, int flags)
{
    if (view == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "NULL view in getbuffer");
        return -1;
    }
    if (self->mat == NULL)
    {
        PyErr_SetString(PyExc_BufferError, "Matrix is not initialized");
        return -1;
    }

    MatrixoxBufferInfo *info = (MatrixoxBufferInfo *)malloc(sizeof(MatrixoxBufferInfo));
    if (!info)
    {
        PyErr_NoMemory();
        return -1;
    }

    int is_c_contig = 0;
    if (self->mat->is_1d)
    {
        view->ndim = 1;
        info->shape[0] = (self->mat->rows == 1) ? self->mat->cols : self->mat->rows;
        info->strides[0] = (self->mat->rows == 1)
                               ? (Py_ssize_t)sizeof(double)
                               : (Py_ssize_t)self->mat->stride * (Py_ssize_t)sizeof(double);
        is_c_contig = (self->mat->rows == 1) || (self->mat->stride == 1);
    }
    else
    {
        view->ndim = 2;
        info->shape[0] = self->mat->rows;
        info->shape[1] = self->mat->cols;
        info->strides[0] = (Py_ssize_t)self->mat->stride * (Py_ssize_t)sizeof(double);
        info->strides[1] = (Py_ssize_t)sizeof(double);
        is_c_contig = (self->mat->stride == self->mat->cols);
    }

    if (((flags & PyBUF_C_CONTIGUOUS) == PyBUF_C_CONTIGUOUS && !is_c_contig) ||
        ((flags & PyBUF_F_CONTIGUOUS) == PyBUF_F_CONTIGUOUS) ||
        ((flags & PyBUF_ANY_CONTIGUOUS) == PyBUF_ANY_CONTIGUOUS && !is_c_contig) ||
        ((flags & PyBUF_STRIDES) != PyBUF_STRIDES && !is_c_contig))
    {
        free(info);
        PyErr_SetString(PyExc_BufferError, "Matrix view is not contiguous");
        return -1;
    }

    view->buf = self->mat->data;
    view->obj = (PyObject *)self;
    Py_INCREF(self);
    view->len = (Py_ssize_t)self->mat->rows * (Py_ssize_t)self->mat->cols * (Py_ssize_t)sizeof(double);
    view->readonly = 0;
    view->itemsize = sizeof(double);
    view->format = (flags & PyBUF_FORMAT) ? "d" : NULL;
    view->shape = (flags & PyBUF_ND) ? info->shape : NULL;
    view->strides = (flags & PyBUF_STRIDES) ? info->strides : NULL;
    view->suboffsets = NULL;
    view->internal = info;
    return 0;
}

static void Matrixox_releasebuffer(Matrixox *self, Py_buffer *view)
{
    (void)self;
    free(view->internal);
    view->internal = NULL;
}

PyBufferProcs Matrixox_as_buffer = {
    .bf_getbuffer = (getbufferproc)Matrixox_getbuffer,
    .bf_releasebuffer = (releasebufferproc)Matrixox_releasebuffer,
};

/* INSTANCE ATTRIBUTES*/
PyMemberDef Matrixox_members[] = {
    {"shape", T_OBJECT_EX, offsetof(Matrixox, shape), READONLY,
     "(rows, cols)"},
    {NULL} /* Sentinel */
};

PyTypeObject MatrixoxType = {
    PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "numox.Matrix",
    .tp_basicsize = sizeof(Matrixox),
    .tp_dealloc = (destructor)Matrixox_dealloc,
    .tp_repr = (reprfunc)Matrixox_repr,
    .tp_as_number = &Matrixox_as_number,
    .tp_as_buffer = &Matrixox_as_buffer,
    .tp_flags = Py_TPFLAGS_DEFAULT |
                Py_TPFLAGS_BASETYPE,
    .tp_doc = "numox.Matrix objects",
    .tp_methods = Matrixox_methods,
    .tp_members = Matrixox_members,
    .tp_as_mapping = &Matrixox_mapping,
    .tp_init = (initproc)Matrixox_init,
    .tp_new = Matrixox_new};

struct PyModuleDef numoxmodule = {
    PyModuleDef_HEAD_INIT,
    "numox",
    "Numc matrix operations",
    -1,
    Matrixox_class_methods};

/* Initialize the numox module */
PyMODINIT_FUNC PyInit_numox(void)
{
    PyObject *m;

    if (PyType_Ready(&MatrixoxType) < 0)
        return NULL;

    m = PyModule_Create(&numoxmodule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&MatrixoxType);
    PyModule_AddObject(m, "Matrix", (PyObject *)&MatrixoxType);
    fflush(stdout);
    return m;
}