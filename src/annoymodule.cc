// Copyright (c) 2013 Spotify AB
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

#include "annoylib.h"
#include "kissrandom.h"
#include "Python.h"
#include "structmember.h"
#include <exception>
#if defined(_MSC_VER) && _MSC_VER == 1500
typedef signed __int32    int32_t;
#else
#include <stdint.h>
#endif


#if defined(ANNOYLIB_USE_AVX512)
#define AVX_INFO "Using 512-bit AVX instructions"
#elif defined(ANNOYLIB_USE_AVX128)
#define AVX_INFO "Using 128-bit AVX instructions"
#else
#define AVX_INFO "Not using AVX instructions"
#endif

#if defined(_MSC_VER)
#define COMPILER_INFO "Compiled using MSC"
#elif defined(__GNUC__)
#define COMPILER_INFO "Compiled on GCC"
#else
#define COMPILER_INFO "Compiled on unknown platform"
#endif

#define ANNOY_DOC (COMPILER_INFO ". " AVX_INFO ".")

#if PY_MAJOR_VERSION >= 3
#define IS_PY3K
#endif

#ifndef Py_TYPE
    #define Py_TYPE(ob) (((PyObject*)(ob))->ob_type)
#endif

#ifdef IS_PY3K
    #define PyInt_FromLong PyLong_FromLong 
#endif

using namespace Annoy;

#ifdef ANNOYLIB_MULTITHREADED_BUILD
  typedef AnnoyIndexMultiThreadedBuildPolicy AnnoyIndexThreadedBuildPolicy;
#else
  typedef AnnoyIndexSingleThreadedBuildPolicy AnnoyIndexThreadedBuildPolicy;
#endif

template class Annoy::AnnoyIndexInterface<int32_t, float>;

class HammingWrapper : public AnnoyIndexInterface<int32_t, float> {
  // Wrapper class for Hamming distance, using composition.
  // This translates binary (float) vectors into packed uint64_t vectors.
  // This is questionable from a performance point of view. Should reconsider this solution.
private:
  int32_t _f_external, _f_internal;
  AnnoyIndex<int32_t, uint64_t, Hamming, Kiss64Random, AnnoyIndexThreadedBuildPolicy> _index;
  void _pack(const float* src, uint64_t* dst) const {
    for (int32_t i = 0; i < _f_internal; i++) {
      dst[i] = 0;
      for (int32_t j = 0; j < 64 && i*64+j < _f_external; j++) {
	dst[i] |= (uint64_t)(src[i * 64 + j] > 0.5) << j;
      }
    }
  };
  void _unpack(const uint64_t* src, float* dst) const {
    for (int32_t i = 0; i < _f_external; i++) {
      dst[i] = (src[i / 64] >> (i % 64)) & 1;
    }
  };
public:
  HammingWrapper(int f) : _f_external(f), _f_internal((f + 63) / 64), _index((f + 63) / 64) {
    _index.set_metadata_identity(f, "hamming");
  };
  bool add_item(int32_t item, const float* w, char**error) {
    vector<uint64_t> w_internal(_f_internal, 0);
    _pack(w, &w_internal[0]);
    return _index.add_item(item, &w_internal[0], error);
  };
  bool build(int q, int n_threads, char** error) { return _index.build(q, n_threads, error); };
  bool unbuild(char** error) { return _index.unbuild(error); };
  bool save(const char* filename, bool prefault, char** error) { return _index.save(filename, prefault, error); };
  bool save_legacy(const char* filename, bool prefault, char** error) { return _index.save_legacy(filename, prefault, error); };
  void unload() { _index.unload(); };
  bool load(const char* filename, bool prefault, char** error) { return _index.load(filename, prefault, error); };
  size_t get_node_size() const { return _index.get_node_size(); };
  const AnnoyMetadata& get_metadata() const { return _index.get_metadata(); };
  bool has_metadata() const { return _index.has_metadata(); };
  const char* get_load_warning() const { return _index.get_load_warning(); };
  const char* get_load_status() const { return _index.get_load_status(); };
  float get_distance(int32_t i, int32_t j) const { return _index.get_distance(i, j); };
  void get_nns_by_item(int32_t item, size_t n, int search_k, vector<int32_t>* result, vector<float>* distances) const {
    if (distances) {
      vector<uint64_t> distances_internal;
      _index.get_nns_by_item(item, n, search_k, result, &distances_internal);
      distances->insert(distances->begin(), distances_internal.begin(), distances_internal.end());
    } else {
      _index.get_nns_by_item(item, n, search_k, result, NULL);
    }
  };
  void get_nns_by_vector(const float* w, size_t n, int search_k, vector<int32_t>* result, vector<float>* distances) const {
    vector<uint64_t> w_internal(_f_internal, 0);
    _pack(w, &w_internal[0]);
    if (distances) {
      vector<uint64_t> distances_internal;
      _index.get_nns_by_vector(&w_internal[0], n, search_k, result, &distances_internal);
      distances->insert(distances->begin(), distances_internal.begin(), distances_internal.end());
    } else {
      _index.get_nns_by_vector(&w_internal[0], n, search_k, result, NULL);
    }
  };
  int32_t get_n_items() const { return _index.get_n_items(); };
  int32_t get_n_trees() const { return _index.get_n_trees(); };
  void verbose(bool v) { _index.verbose(v); };
  void get_item(int32_t item, float* v) const {
    vector<uint64_t> v_internal(_f_internal, 0);
    _index.get_item(item, &v_internal[0]);
    _unpack(&v_internal[0], v);
  };
  void set_seed(uint64_t q) { _index.set_seed(q); };
  bool on_disk_build(const char* filename, char** error) { return _index.on_disk_build(filename, error); };
};

// annoy python object
typedef struct {
  PyObject_HEAD
  int f;
  AnnoyIndexInterface<int32_t, float>* ptr;
} py_annoy;

static size_t expected_node_size(int f, const char* metric) {
  if (!metric || f <= 0)
    return 0;
  typedef Angular::Node<int32_t, float> AngularNode;
  typedef Minkowski::Node<int32_t, float> MinkowskiNode;
  typedef DotProduct::Node<int32_t, float> DotProductNode;
  typedef Hamming::Node<int32_t, uint64_t> HammingNode;
  if (!strcmp(metric, "angular"))
    return offsetof(AngularNode, v) + f * sizeof(float);
  if (!strcmp(metric, "euclidean") || !strcmp(metric, "manhattan"))
    return offsetof(MinkowskiNode, v) + f * sizeof(float);
  if (!strcmp(metric, "dot"))
    return offsetof(DotProductNode, v) + f * sizeof(float);
  if (!strcmp(metric, "hamming")) {
    int internal_f = (f + 63) / 64;
    return offsetof(HammingNode, v) + internal_f * sizeof(uint64_t);
  }
  return 0;
}

static int metadata_dict_set(PyObject* dict, const char* key, PyObject* value) {
  if (!value)
    return -1;
  int result = PyDict_SetItemString(dict, key, value);
  Py_DECREF(value);
  return result;
}

static void close_metadata_fd(int fd) {
#ifndef _MSC_VER
  close(fd);
#else
  _close(fd);
#endif
}

static PyObject* metadata_to_python(
    const AnnoyMetadata& metadata, bool present, bool counts_known,
    const char* path, const char* status, const char* warning, const char* error,
    size_t file_size, int caller_dimension, const char* caller_metric) {
  PyObject* result = PyDict_New();
  if (!result)
    return NULL;

  PyObject* path_obj = PyUnicode_FromString(path);
  PyObject* status_obj = PyUnicode_FromString(status ? status : "");
  const char* format_name = present ? "annoy-v2" :
                            (status && !strcmp(status, "legacy") ? "legacy" : "unknown");
  PyObject* format_obj = PyUnicode_FromString(format_name);
  PyObject* present_obj = PyBool_FromLong(present ? 1 : 0);
  PyObject* dimension_obj = metadata.dimension > 0
                              ? PyLong_FromLong(metadata.dimension) : Py_None;
  if (dimension_obj == Py_None)
    Py_INCREF(Py_None);
  PyObject* metric_obj = metadata.metric.empty()
                           ? Py_None : PyUnicode_FromString(metadata.metric.c_str());
  if (metric_obj == Py_None)
    Py_INCREF(Py_None);
  PyObject* node_size_obj = metadata.node_size
                              ? PyLong_FromUnsignedLong(metadata.node_size) : Py_None;
  if (node_size_obj == Py_None)
    Py_INCREF(Py_None);
  PyObject* item_count_obj = counts_known
                               ? PyLong_FromUnsignedLongLong(metadata.item_count) : Py_None;
  if (item_count_obj == Py_None)
    Py_INCREF(Py_None);
  PyObject* tree_count_obj = counts_known
                               ? PyLong_FromUnsignedLong(metadata.tree_count) : Py_None;
  if (tree_count_obj == Py_None)
    Py_INCREF(Py_None);
  PyObject* n_items_obj = counts_known
                            ? PyLong_FromUnsignedLongLong(metadata.item_count) : Py_None;
  if (n_items_obj == Py_None)
    Py_INCREF(Py_None);
  PyObject* n_trees_obj = counts_known
                            ? PyLong_FromUnsignedLong(metadata.tree_count) : Py_None;
  if (n_trees_obj == Py_None)
    Py_INCREF(Py_None);
  PyObject* body_size_obj = PyLong_FromUnsignedLongLong(
      present ? metadata.body_size : (uint64_t)file_size);
  PyObject* file_size_obj = PyLong_FromUnsignedLongLong((uint64_t)file_size);
  PyObject* version_obj = PyLong_FromUnsignedLong(metadata.format_version);
  PyObject* seed_obj = PyLong_FromUnsignedLongLong(metadata.seed);
  PyObject* writer_obj = metadata.writer_version.empty()
                           ? Py_None : PyUnicode_FromString(metadata.writer_version.c_str());
  if (writer_obj == Py_None)
    Py_INCREF(Py_None);
  PyObject* warning_obj = warning && warning[0]
                            ? PyUnicode_FromString(warning) : Py_None;
  if (warning_obj == Py_None)
    Py_INCREF(Py_None);
  PyObject* error_obj = error && error[0]
                          ? PyUnicode_FromString(error) : Py_None;
  if (error_obj == Py_None)
    Py_INCREF(Py_None);
  PyObject* caller_dimension_obj = caller_dimension > 0
                                     ? PyLong_FromLong(caller_dimension) : Py_None;
  if (caller_dimension_obj == Py_None)
    Py_INCREF(Py_None);
  PyObject* caller_metric_obj = caller_metric
                                  ? PyUnicode_FromString(caller_metric) : Py_None;
  if (caller_metric_obj == Py_None)
    Py_INCREF(Py_None);

  int ok = metadata_dict_set(result, "path", path_obj) == 0 &&
           metadata_dict_set(result, "status", status_obj) == 0 &&
           metadata_dict_set(result, "format", format_obj) == 0 &&
           metadata_dict_set(result, "format_version", version_obj) == 0 &&
           metadata_dict_set(result, "metadata_present", present_obj) == 0 &&
           metadata_dict_set(result, "dimension", dimension_obj) == 0 &&
           metadata_dict_set(result, "metric", metric_obj) == 0 &&
           metadata_dict_set(result, "node_size", node_size_obj) == 0 &&
           metadata_dict_set(result, "item_count", item_count_obj) == 0 &&
           metadata_dict_set(result, "items", item_count_obj) == 0 &&
           metadata_dict_set(result, "n_items", n_items_obj) == 0 &&
           metadata_dict_set(result, "tree_count", tree_count_obj) == 0 &&
           metadata_dict_set(result, "trees", tree_count_obj) == 0 &&
           metadata_dict_set(result, "n_trees", n_trees_obj) == 0 &&
           metadata_dict_set(result, "body_size", body_size_obj) == 0 &&
           metadata_dict_set(result, "file_size", file_size_obj) == 0 &&
           metadata_dict_set(result, "seed", seed_obj) == 0 &&
           metadata_dict_set(result, "writer_version", writer_obj) == 0 &&
           metadata_dict_set(result, "warning", warning_obj) == 0 &&
           metadata_dict_set(result, "error", error_obj) == 0 &&
           metadata_dict_set(result, "caller_dimension", caller_dimension_obj) == 0 &&
           metadata_dict_set(result, "caller_metric", caller_metric_obj) == 0;
  if (!ok) {
    Py_DECREF(result);
    return NULL;
  }
  return result;
}


static PyObject *
py_an_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
  py_annoy *self = (py_annoy *)type->tp_alloc(type, 0);
  if (self == NULL) {
    return NULL;
  }
  const char *metric = NULL;

  static char const * kwlist[] = {"f", "metric", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i|s", (char**)kwlist, &self->f, &metric))
    return NULL;
  if (!metric) {
    // This keeps coming up, see #368 etc
    PyErr_WarnEx(PyExc_FutureWarning, "The default argument for metric will be removed "
		 "in future version of Annoy. Please pass metric='angular' explicitly.", 1);
    self->ptr = new AnnoyIndex<int32_t, float, Angular, Kiss64Random, AnnoyIndexThreadedBuildPolicy>(self->f);
  } else if (!strcmp(metric, "angular")) {
   self->ptr = new AnnoyIndex<int32_t, float, Angular, Kiss64Random, AnnoyIndexThreadedBuildPolicy>(self->f);
  } else if (!strcmp(metric, "euclidean")) {
    self->ptr = new AnnoyIndex<int32_t, float, Euclidean, Kiss64Random, AnnoyIndexThreadedBuildPolicy>(self->f);
  } else if (!strcmp(metric, "manhattan")) {
    self->ptr = new AnnoyIndex<int32_t, float, Manhattan, Kiss64Random, AnnoyIndexThreadedBuildPolicy>(self->f);
  } else if (!strcmp(metric, "hamming")) {
    self->ptr = new HammingWrapper(self->f);
  } else if (!strcmp(metric, "dot")) {
    self->ptr = new AnnoyIndex<int32_t, float, DotProduct, Kiss64Random, AnnoyIndexThreadedBuildPolicy>(self->f);
  } else {
    PyErr_SetString(PyExc_ValueError, "No such metric");
    return NULL;
  }

  return (PyObject *)self;
}


static int 
py_an_init(py_annoy *self, PyObject *args, PyObject *kwargs) {
  // Seems to be needed for Python 3
  const char *metric = NULL;
  int f;
  static char const * kwlist[] = {"f", "metric", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i|s", (char**)kwlist, &f, &metric))
    return (int) NULL;
  return 0;
}


static void 
py_an_dealloc(py_annoy* self) {
  delete self->ptr;
  Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyMemberDef py_annoy_members[] = {
  {(char*)"f", T_INT, offsetof(py_annoy, f), 0,
   (char*)""},
  {NULL}	/* Sentinel */
};


static PyObject *
py_an_load(py_annoy *self, PyObject *args, PyObject *kwargs) {
  char *filename, *error;
  bool prefault = false;
  if (!self->ptr) 
    return NULL;
  static char const * kwlist[] = {"fn", "prefault", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|b", (char**)kwlist, &filename, &prefault))
    return NULL;

  if (!self->ptr->load(filename, prefault, &error)) {
    PyErr_SetString(PyExc_IOError, error);
    free(error);
    return NULL;
  }
  const char* warning = self->ptr->get_load_warning();
  if (warning && warning[0] && PyErr_WarnEx(PyExc_RuntimeWarning, warning, 1) < 0)
    return NULL;
  Py_RETURN_TRUE;
}


static PyObject *
py_an_save(py_annoy *self, PyObject *args, PyObject *kwargs) {
  char *filename, *error;
  bool prefault = false;
  int legacy = 0;
  const char* format = NULL;
  if (!self->ptr) 
    return NULL;
  static char const * kwlist[] = {"fn", "prefault", "legacy", "format", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|bis", (char**)kwlist,
                                   &filename, &prefault, &legacy, &format))
    return NULL;
  if (format) {
    if (!strcmp(format, "legacy") || !strcmp(format, "v1"))
      legacy = 1;
    else if (!strcmp(format, "current") || !strcmp(format, "v2"))
      legacy = 0;
    else {
      PyErr_SetString(PyExc_ValueError, "format must be 'legacy' or 'current'");
      return NULL;
    }
  }

  bool saved = legacy
                 ? self->ptr->save_legacy(filename, prefault, &error)
                 : self->ptr->save(filename, prefault, &error);
  if (!saved) {
    PyErr_SetString(PyExc_IOError, error);
    free(error);
    return NULL;
  }
  Py_RETURN_TRUE;
}

static PyObject *
py_an_save_legacy(py_annoy *self, PyObject *args, PyObject *kwargs) {
  char *filename, *error;
  bool prefault = false;
  if (!self->ptr)
    return NULL;
  static char const * kwlist[] = {"fn", "prefault", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|b", (char**)kwlist,
                                   &filename, &prefault))
    return NULL;
  if (!self->ptr->save_legacy(filename, prefault, &error)) {
    PyErr_SetString(PyExc_IOError, error);
    free(error);
    return NULL;
  }
  Py_RETURN_TRUE;
}

static PyObject *
py_an_get_metadata(py_annoy *self, PyObject *args) {
  if (!self->ptr)
    return NULL;
  const AnnoyMetadata& metadata = self->ptr->get_metadata();
  return metadata_to_python(
      metadata, self->ptr->has_metadata(), true, "", self->ptr->get_load_status(),
      self->ptr->get_load_warning(), NULL, 0, -1, NULL);
}

static PyObject *
py_an_get_load_status(py_annoy *self, PyObject *args) {
  if (!self->ptr)
    return NULL;
  return PyUnicode_FromString(self->ptr->get_load_status());
}

static PyObject *
py_an_get_load_warning(py_annoy *self, PyObject *args) {
  if (!self->ptr)
    return NULL;
  return PyUnicode_FromString(self->ptr->get_load_warning());
}

static PyObject *
py_read_index_metadata(PyObject *self, PyObject *args, PyObject *kwargs) {
  char* filename;
  int f = -1;
  const char* metric = NULL;
  static char const * kwlist[] = {"fn", "f", "metric", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|is", (char**)kwlist,
                                   &filename, &f, &metric))
    return NULL;

  AnnoyMetadata metadata = AnnoyMetadata();
  const char* status = "ok";
  char warning[768];
  char error_message[1024];
  warning[0] = '\0';
  error_message[0] = '\0';
  size_t file_size = 0;
  size_t expected_size = expected_node_size(f, metric);
  int fd = -1;
  int io_error = 0;
  off_t raw_size = -1;
  size_t header_bytes = 0;
  size_t tail_bytes = 0;
  bool read_ok = false;
  bool has_trailer = false;
  uint8_t header[ANNOY_METADATA_HEADER_SIZE];
  uint8_t tail[16];
  memset(header, 0, sizeof(header));
  memset(tail, 0, sizeof(tail));

  Py_BEGIN_ALLOW_THREADS
  {
#ifndef _MSC_VER
    fd = open(filename, O_RDONLY, (int)0400);
#else
    fd = _open(filename, _O_RDONLY, (int)0400);
#endif
    if (fd == -1) {
      io_error = errno;
    } else {
      raw_size = lseek_getsize(fd);
      if (raw_size >= 0) {
        file_size = (size_t)raw_size;
        header_bytes = file_size < sizeof(header) ? file_size : sizeof(header);
        if (file_size > 0 &&
            annoy_lseek(fd, 0, SEEK_SET) != (off_t)-1 &&
            annoy_read(fd, header, header_bytes) == (off_t)header_bytes) {
          read_ok = true;
          tail_bytes = file_size < sizeof(tail) ? file_size : sizeof(tail);
          if (tail_bytes > 0 &&
              annoy_lseek(fd, raw_size - (off_t)tail_bytes, SEEK_SET) != (off_t)-1 &&
              annoy_read(fd, tail, tail_bytes) == (off_t)tail_bytes)
            has_trailer = metadata_has_trailer_magic(tail, tail_bytes);
        }
      }
      io_error = errno;
      close_metadata_fd(fd);
    }
  }
  Py_END_ALLOW_THREADS

  if (fd == -1) {
    status = "unreadable";
    snprintf(error_message, sizeof(error_message),
             "Unable to open index '%s': %s (%d)", filename, strerror(io_error), io_error);
    return metadata_to_python(metadata, false, false, filename, status, NULL,
                              error_message, 0, f, metric);
  }

  if (raw_size < 0) {
    status = "unreadable";
    snprintf(error_message, sizeof(error_message),
             "Unable to get size of index '%s': %s (%d)",
             filename, strerror(io_error), io_error);
    return metadata_to_python(metadata, false, false, filename, status, NULL,
                              error_message, 0, f, metric);
  }
  if (file_size == 0) {
    status = "corrupted";
    snprintf(error_message, sizeof(error_message),
             "Corrupted index '%s': file is empty", filename);
    return metadata_to_python(metadata, false, false, filename, status, NULL,
                              error_message, 0, f, metric);
  }

  if (!read_ok) {
    status = "unreadable";
    snprintf(error_message, sizeof(error_message),
             "Unable to read index header '%s': %s (%d)",
             filename, strerror(io_error), io_error);
    return metadata_to_python(metadata, false, false, filename, status, NULL,
                              error_message, file_size, f, metric);
  }

  if (metadata_has_magic(header, header_bytes)) {
    std::string parse_error;
    std::string parse_kind;
    if (!metadata_parse_header(header, header_bytes, &metadata,
                               &parse_error, &parse_kind)) {
      status = parse_kind == "unsupported" ? "unsupported" : "corrupted";
      snprintf(error_message, sizeof(error_message),
               "%s index '%s': %s",
               parse_kind == "unsupported" ? "Unsupported" : "Corrupted",
               filename, parse_error.c_str());
      return metadata_to_python(metadata, false, false, filename, status, NULL,
                                error_message, file_size, f, metric);
    }
    uint64_t declared_size = (uint64_t)ANNOY_METADATA_HEADER_SIZE +
                             metadata.body_size + metadata.trailer_size;
    if ((uint64_t)file_size != declared_size) {
      status = "corrupted";
      snprintf(error_message, sizeof(error_message),
               "Corrupted index '%s': header declares %llu bytes, actual %llu bytes",
               filename, (unsigned long long)declared_size,
               (unsigned long long)file_size);
      return metadata_to_python(metadata, true, true, filename, status, NULL,
                                error_message, file_size, f, metric);
    }
    if (!has_trailer) {
      status = "corrupted";
      snprintf(error_message, sizeof(error_message),
               "Corrupted index '%s': metadata header is valid but the format trailer is missing",
               filename);
      return metadata_to_python(metadata, true, true, filename, status, NULL,
                                error_message, file_size, f, metric);
    }

    bool compatible = true;
    if (f > 0 && metadata.dimension != f)
      compatible = false;
    if (metric && metadata.metric != metric)
      compatible = false;
    if (expected_size && metadata.node_size != expected_size)
      compatible = false;
    if (!compatible) {
      status = "incompatible";
      snprintf(error_message, sizeof(error_message),
               "Index parameter mismatch for '%s': caller dimension=%d metric='%s' "
               "node_size=%zu; file dimension=%d metric='%s' node_size=%u",
               filename, f, metric ? metric : "(not provided)",
               expected_size, metadata.dimension, metadata.metric.c_str(),
               metadata.node_size);
      return metadata_to_python(metadata, true, true, filename, status, NULL,
                                error_message, file_size, f, metric);
    }
    return metadata_to_python(metadata, true, true, filename, "ok", NULL,
                              NULL, file_size, f, metric);
  }

  metadata.format_version = 0;
  metadata.body_size = (uint64_t)file_size;
  metadata.writer_version = "legacy";
  status = "legacy";
  if (has_trailer) {
    status = "corrupted";
    snprintf(error_message, sizeof(error_message),
             "Corrupted index '%s': metadata header magic is damaged but the "
             "new-format trailer is present",
             filename);
  } else if (expected_size) {
    metadata.dimension = f;
    metadata.metric = metric;
    metadata.node_size = (uint32_t)expected_size;
    if (file_size % expected_size != 0) {
      status = "corrupted";
      snprintf(error_message, sizeof(error_message),
               "Corrupted index '%s': size %llu is not a multiple of caller node size %zu "
               "(dimension=%d, metric='%s')",
               filename, (unsigned long long)file_size, expected_size, f, metric);
    } else {
      snprintf(warning, sizeof(warning),
               "Legacy index '%s' has no parameter metadata; compatibility is inferred "
               "from caller dimension=%d, metric='%s', node_size=%zu. Rebuild to add a fingerprint.",
               filename, f, metric, expected_size);
    }
  } else {
    snprintf(warning, sizeof(warning),
             "Legacy index '%s' has no parameter metadata; dimension and metric are unknown. "
             "Provide f and metric to validate compatibility.",
             filename);
  }
  return metadata_to_python(metadata, false, false, filename, status,
                            warning, error_message, file_size, f, metric);
}


PyObject*
get_nns_to_python(const vector<int32_t>& result, const vector<float>& distances, int include_distances) {
  PyObject* l = NULL;
  PyObject* d = NULL;
  PyObject* t = NULL;

  if ((l = PyList_New(result.size())) == NULL) {
    goto error;
  }
  for (size_t i = 0; i < result.size(); i++) {
    PyObject* res = PyInt_FromLong(result[i]);
    if (res == NULL) {
      goto error;
    }
    PyList_SetItem(l, i, res);
  }
  if (!include_distances)
    return l;

  if ((d = PyList_New(distances.size())) == NULL) {
    goto error;
  }

  for (size_t i = 0; i < distances.size(); i++) {
    PyObject* dist = PyFloat_FromDouble(distances[i]);
    if (dist == NULL) {
      goto error;
    }
    PyList_SetItem(d, i, dist);
  }

  if ((t = PyTuple_Pack(2, l, d)) == NULL) {
    goto error;
  }
  Py_XDECREF(l);
  Py_XDECREF(d);

  return t;

  error:
    Py_XDECREF(l);
    Py_XDECREF(d);
    Py_XDECREF(t);
    return NULL;
}


bool check_constraints(py_annoy *self, int32_t item, bool building) {
  if (item < 0) {
    PyErr_SetString(PyExc_IndexError, "Item index can not be negative");
    return false;
  } else if (!building && item >= self->ptr->get_n_items()) {
    PyErr_SetString(PyExc_IndexError, "Item index larger than the largest item index");
    return false;
  } else {
    return true;
  }
}

static PyObject* 
py_an_get_nns_by_item(py_annoy *self, PyObject *args, PyObject *kwargs) {
  int32_t item, n, search_k=-1, include_distances=0;
  if (!self->ptr) 
    return NULL;

  static char const * kwlist[] = {"i", "n", "search_k", "include_distances", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii|ii", (char**)kwlist, &item, &n, &search_k, &include_distances))
    return NULL;

  if (!check_constraints(self, item, false)) {
    return NULL;
  }

  vector<int32_t> result;
  vector<float> distances;

  Py_BEGIN_ALLOW_THREADS;
  self->ptr->get_nns_by_item(item, n, search_k, &result, include_distances ? &distances : NULL);
  Py_END_ALLOW_THREADS;

  return get_nns_to_python(result, distances, include_distances);
}


bool
convert_list_to_vector(PyObject* v, int f, vector<float>* w) {
  Py_ssize_t length = PyObject_Size(v);
  if (length == -1) {
    return false;
  }
  if (length != f) {
    PyErr_Format(PyExc_IndexError, "Vector has wrong length (expected %d, got %ld)", f, length);
    return false;
  }

  for (int z = 0; z < f; z++) {
    PyObject *key = PyInt_FromLong(z);
    if (key == NULL) {
      return false;
    }
    PyObject *pf = PyObject_GetItem(v, key);
    Py_DECREF(key);
    if (pf == NULL) {
      return false;
    }
    double value = PyFloat_AsDouble(pf);
    Py_DECREF(pf);
    if (value == -1.0 && PyErr_Occurred()) {
      return false;
    }
    (*w)[z] = value;
  }
  return true;
}

static PyObject* 
py_an_get_nns_by_vector(py_annoy *self, PyObject *args, PyObject *kwargs) {
  PyObject* v;
  int32_t n, search_k=-1, include_distances=0;
  if (!self->ptr) 
    return NULL;

  static char const * kwlist[] = {"vector", "n", "search_k", "include_distances", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi|ii", (char**)kwlist, &v, &n, &search_k, &include_distances))
    return NULL;

  vector<float> w(self->f);
  if (!convert_list_to_vector(v, self->f, &w)) {
    return NULL;
  }

  vector<int32_t> result;
  vector<float> distances;

  Py_BEGIN_ALLOW_THREADS;
  self->ptr->get_nns_by_vector(&w[0], n, search_k, &result, include_distances ? &distances : NULL);
  Py_END_ALLOW_THREADS;

  return get_nns_to_python(result, distances, include_distances);
}


static PyObject* 
py_an_get_item_vector(py_annoy *self, PyObject *args) {
  int32_t item;
  if (!self->ptr) 
    return NULL;
  if (!PyArg_ParseTuple(args, "i", &item))
    return NULL;

  if (!check_constraints(self, item, false)) {
    return NULL;
  }

  vector<float> v(self->f);
  self->ptr->get_item(item, &v[0]);
  PyObject* l = PyList_New(self->f);
  if (l == NULL) {
    return NULL;
  }
  for (int z = 0; z < self->f; z++) {
    PyObject* dist = PyFloat_FromDouble(v[z]);
    if (dist == NULL) {
      goto error;
    }
    PyList_SetItem(l, z, dist);
  }

  return l;

  error:
    Py_XDECREF(l);
    return NULL;
}


static PyObject* 
py_an_add_item(py_annoy *self, PyObject *args, PyObject* kwargs) {
  PyObject* v;
  int32_t item;
  if (!self->ptr) 
    return NULL;
  static char const * kwlist[] = {"i", "vector", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iO", (char**)kwlist, &item, &v))
    return NULL;

  if (!check_constraints(self, item, true)) {
    return NULL;
  }

  vector<float> w(self->f);
  if (!convert_list_to_vector(v, self->f, &w)) {
    return NULL;
  }
  char* error;
  if (!self->ptr->add_item(item, &w[0], &error)) {
    PyErr_SetString(PyExc_Exception, error);
    free(error);
    return NULL;
  }

  Py_RETURN_NONE;
}

static PyObject *
py_an_on_disk_build(py_annoy *self, PyObject *args, PyObject *kwargs) {
  char *filename, *error;
  if (!self->ptr)
    return NULL;
  static char const * kwlist[] = {"fn", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s", (char**)kwlist, &filename))
    return NULL;

  if (!self->ptr->on_disk_build(filename, &error)) {
    PyErr_SetString(PyExc_IOError, error);
    free(error);
    return NULL;
  }
  Py_RETURN_TRUE;
}

static PyObject *
py_an_build(py_annoy *self, PyObject *args, PyObject *kwargs) {
  int q;
  int n_jobs = -1;
  if (!self->ptr) 
    return NULL;
  static char const * kwlist[] = {"n_trees", "n_jobs", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i|i", (char**)kwlist, &q, &n_jobs))
    return NULL;

  bool res;
  char* error;
  Py_BEGIN_ALLOW_THREADS;
  res = self->ptr->build(q, n_jobs, &error);
  Py_END_ALLOW_THREADS;
  if (!res) {
    PyErr_SetString(PyExc_Exception, error);
    free(error);
    return NULL;
  }

  Py_RETURN_TRUE;
}


static PyObject *
py_an_unbuild(py_annoy *self) {
  if (!self->ptr) 
    return NULL;

  char* error;
  if (!self->ptr->unbuild(&error)) {
    PyErr_SetString(PyExc_Exception, error);
    free(error);
    return NULL;
  }

  Py_RETURN_TRUE;
}


static PyObject *
py_an_unload(py_annoy *self) {
  if (!self->ptr) 
    return NULL;

  self->ptr->unload();

  Py_RETURN_TRUE;
}


static PyObject *
py_an_get_distance(py_annoy *self, PyObject *args) {
  int32_t i, j;
  if (!self->ptr) 
    return NULL;
  if (!PyArg_ParseTuple(args, "ii", &i, &j))
    return NULL;

  if (!check_constraints(self, i, false) || !check_constraints(self, j, false)) {
    return NULL;
  }

  double d = self->ptr->get_distance(i,j);
  return PyFloat_FromDouble(d);
}


static PyObject *
py_an_get_n_items(py_annoy *self) {
  if (!self->ptr) 
    return NULL;

  int32_t n = self->ptr->get_n_items();
  return PyInt_FromLong(n);
}

static PyObject *
py_an_get_n_trees(py_annoy *self) {
  if (!self->ptr) 
    return NULL;

  int32_t n = self->ptr->get_n_trees();
  return PyInt_FromLong(n);
}

static PyObject *
py_an_verbose(py_annoy *self, PyObject *args) {
  int verbose;
  if (!self->ptr) 
    return NULL;
  if (!PyArg_ParseTuple(args, "i", &verbose))
    return NULL;

  self->ptr->verbose((bool)verbose);

  Py_RETURN_TRUE;
}


static PyObject *
py_an_set_seed(py_annoy *self, PyObject *args) {
  int q;
  if (!self->ptr)
    return NULL;
  if (!PyArg_ParseTuple(args, "i", &q))
    return NULL;

  self->ptr->set_seed(q);

  Py_RETURN_NONE;
}


static PyMethodDef AnnoyMethods[] = {
  {"load",	(PyCFunction)py_an_load, METH_VARARGS | METH_KEYWORDS, "Loads (mmaps) an index from disk."},
  {"save",	(PyCFunction)py_an_save, METH_VARARGS | METH_KEYWORDS, "Saves the index to disk."},
  {"save_legacy",(PyCFunction)py_an_save_legacy, METH_VARARGS | METH_KEYWORDS, "Saves the index in the legacy body-only format."},
  {"get_metadata",(PyCFunction)py_an_get_metadata, METH_NOARGS, "Returns metadata for the currently loaded index."},
  {"get_load_status",(PyCFunction)py_an_get_load_status, METH_NOARGS, "Returns the compatibility status from the last load."},
  {"get_load_warning",(PyCFunction)py_an_get_load_warning, METH_NOARGS, "Returns the compatibility warning from the last load."},
  {"get_nns_by_item",(PyCFunction)py_an_get_nns_by_item, METH_VARARGS | METH_KEYWORDS, "Returns the `n` closest items to item `i`.\n\n:param search_k: the query will inspect up to `search_k` nodes.\n`search_k` gives you a run-time tradeoff between better accuracy and speed.\n`search_k` defaults to `n_trees * n` if not provided.\n\n:param include_distances: If `True`, this function will return a\n2 element tuple of lists. The first list contains the `n` closest items.\nThe second list contains the corresponding distances."},
  {"get_nns_by_vector",(PyCFunction)py_an_get_nns_by_vector, METH_VARARGS | METH_KEYWORDS, "Returns the `n` closest items to vector `vector`.\n\n:param search_k: the query will inspect up to `search_k` nodes.\n`search_k` gives you a run-time tradeoff between better accuracy and speed.\n`search_k` defaults to `n_trees * n` if not provided.\n\n:param include_distances: If `True`, this function will return a\n2 element tuple of lists. The first list contains the `n` closest items.\nThe second list contains the corresponding distances."},
  {"get_item_vector",(PyCFunction)py_an_get_item_vector, METH_VARARGS, "Returns the vector for item `i` that was previously added."},
  {"add_item",(PyCFunction)py_an_add_item, METH_VARARGS | METH_KEYWORDS, "Adds item `i` (any nonnegative integer) with vector `v`.\n\nNote that it will allocate memory for `max(i)+1` items."},
  {"on_disk_build",(PyCFunction)py_an_on_disk_build, METH_VARARGS | METH_KEYWORDS, "Build will be performed with storage on disk instead of RAM."},
  {"build",(PyCFunction)py_an_build, METH_VARARGS | METH_KEYWORDS, "Builds a forest of `n_trees` trees.\n\nMore trees give higher precision when querying. After calling `build`,\nno more items can be added. `n_jobs` specifies the number of threads used to build the trees. `n_jobs=-1` uses all available CPU cores."},
  {"unbuild",(PyCFunction)py_an_unbuild, METH_NOARGS, "Unbuilds the tree in order to allows adding new items.\n\nbuild() has to be called again afterwards in order to\nrun queries."},
  {"unload",(PyCFunction)py_an_unload, METH_NOARGS, "Unloads an index from disk."},
  {"get_distance",(PyCFunction)py_an_get_distance, METH_VARARGS, "Returns the distance between items `i` and `j`."},
  {"get_n_items",(PyCFunction)py_an_get_n_items, METH_NOARGS, "Returns the number of items in the index."},
  {"get_n_trees",(PyCFunction)py_an_get_n_trees, METH_NOARGS, "Returns the number of trees in the index."},
  {"verbose",(PyCFunction)py_an_verbose, METH_VARARGS, ""},
  {"set_seed",(PyCFunction)py_an_set_seed, METH_VARARGS, "Sets the seed of Annoy's random number generator."},
  {NULL, NULL, 0, NULL}		 /* Sentinel */
};


static PyTypeObject PyAnnoyType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "annoy.Annoy",          /*tp_name*/
  sizeof(py_annoy),       /*tp_basicsize*/
  0,                      /*tp_itemsize*/
  (destructor)py_an_dealloc, /*tp_dealloc*/
  0,                      /*tp_print*/
  0,                      /*tp_getattr*/
  0,                      /*tp_setattr*/
  0,                      /*tp_compare*/
  0,                      /*tp_repr*/
  0,                      /*tp_as_number*/
  0,                      /*tp_as_sequence*/
  0,                      /*tp_as_mapping*/
  0,                      /*tp_hash */
  0,                      /*tp_call*/
  0,                      /*tp_str*/
  0,                      /*tp_getattro*/
  0,                      /*tp_setattro*/
  0,                      /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
  ANNOY_DOC,              /* tp_doc */
  0,                      /* tp_traverse */
  0,                      /* tp_clear */
  0,                      /* tp_richcompare */
  0,                      /* tp_weaklistoffset */
  0,                      /* tp_iter */
  0,                      /* tp_iternext */
  AnnoyMethods,           /* tp_methods */
  py_annoy_members,       /* tp_members */
  0,                      /* tp_getset */
  0,                      /* tp_base */
  0,                      /* tp_dict */
  0,                      /* tp_descr_get */
  0,                      /* tp_descr_set */
  0,                      /* tp_dictoffset */
  (initproc)py_an_init,   /* tp_init */
  0,                      /* tp_alloc */
  py_an_new,              /* tp_new */
};

static PyMethodDef module_methods[] = {
  {"read_index_metadata",(PyCFunction)py_read_index_metadata, METH_VARARGS | METH_KEYWORDS,
   "Reads index metadata from the file header without loading the index body."},
  {NULL}	/* Sentinel */
};

#if PY_MAJOR_VERSION >= 3
  static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "annoylib",          /* m_name */
    ANNOY_DOC,           /* m_doc */
    -1,                  /* m_size */
    module_methods,      /* m_methods */
    NULL,                /* m_reload */
    NULL,                /* m_traverse */
    NULL,                /* m_clear */
    NULL,                /* m_free */
  };
#endif

PyObject *create_module(void) {
  PyObject *m;

  if (PyType_Ready(&PyAnnoyType) < 0)
    return NULL;

#if PY_MAJOR_VERSION >= 3
  m = PyModule_Create(&moduledef);
#else
  m = Py_InitModule("annoylib", module_methods);
#endif

  if (m == NULL)
    return NULL;

  Py_INCREF(&PyAnnoyType);
  PyModule_AddObject(m, "Annoy", (PyObject *)&PyAnnoyType);
  return m;
}

#if PY_MAJOR_VERSION >= 3
  PyMODINIT_FUNC PyInit_annoylib(void) {
    return create_module();      // it should return moudule object in py3
  }
#else
  PyMODINIT_FUNC initannoylib(void) {
    create_module();
  }
#endif


// vim: tabstop=2 shiftwidth=2
