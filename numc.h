#include "matrix.h"

/*
 * Defines the struct that represents the object
 * Has the default PyObject_HEAD so it can be a python object
 * It also has the matrix that is being wrapped
 * is of type PyObject
 */
typedef struct {
    PyObject_HEAD
    matrix* mat;
    PyObject *shape;
} Matrixox;

/* Function definitions */
int init_rand(PyObject *self, int rows, int cols, unsigned int seed, double low, double high);
int init_fill(PyObject *self, int rows, int cols, double val);
int init_1d(PyObject *self, int rows, int cols, PyObject *lst);
int init_2d(PyObject *self, PyObject *lst);
void Matrixox_dealloc(Matrixox *self);
PyObject *Matrixox_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
int Matrixox_init(PyObject *self, PyObject *args, PyObject *kwds);
PyObject *Matrixox_to_list(Matrixox *self);
PyObject *Matrixox_repr(PyObject *self);
PyObject *Matrixox_set_value(Matrixox *self, PyObject* args);
PyObject *Matrixox_get_value(Matrixox *self, PyObject* args);
PyObject *Matrixox_add(Matrixox* self, PyObject* args);
PyObject *Matrixox_sub(Matrixox* self, PyObject* args);
PyObject *Matrixox_multiply(Matrixox* self, PyObject *args);
PyObject *Matrixox_neg(Matrixox* self);
PyObject *Matrixox_abs(Matrixox *self);
PyObject *Matrixox_pow(Matrixox *self, PyObject *pow, PyObject *optional);

