#define Py_LIMITED_API 0x030A0000
#include <Python.h>

// 导入 ascend_add_pipeline._C 时，动态库被加载，静态注册代码随之执行。
// 这里的空模块仅提供 Python 扩展入口；真正的算子通过 TORCH_LIBRARY 注册。
extern "C" PyObject* PyInit__C() {
    static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_C", nullptr, -1, nullptr};
    return PyModule_Create(&module);
}
