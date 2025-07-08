# Valgrind Memory Analysis Guide for C/C++ Programmers

## Overview

This guide demonstrates how to use Valgrind to detect memory errors in C/C++ code, based on real bugs found in the nnstreamer Tizen HAL implementations. Each bug pattern includes the problematic code, Valgrind output, and the correct fix.

## Quick Start

```bash
# Compile the demo program
g++ -o memory_leak_demo memory_leak_demo.cpp -std=c++11 -g

# Run with Valgrind
valgrind --tool=memcheck --leak-check=full --track-origins=yes --show-leak-kinds=all ./memory_leak_demo
```

## Bug Pattern 1: Constructor Exception Memory Leaks

### 🚨 The Problem

When a constructor throws an exception after allocating resources, the destructor is **never called**, causing memory leaks.

**Real bug from `tensor_filter_tizen_hal_snpe.cc:73`:**
```cpp
snpe_tizen_hal_subplugin::snpe_tizen_hal_subplugin ()
    : tensor_filter_subplugin (), hal_handle (nullptr)
{
  int ret = hal_ml_create ("snpe", &hal_handle);  // Allocates memory
  if (ret == HAL_ML_ERROR_INVALID_PARAMETER) {
    throw std::invalid_argument ("SNPE HAL is not supported");  // ❌ LEAK!
  }
  // Destructor is NEVER called when constructor throws
}
```

### 📊 Valgrind Output
```
==2299== 100 bytes in 1 blocks are definitely lost in loss record 1 of 1
==2299==    at 0x484A858: malloc (in /usr/libexec/valgrind/vgpreload_memcheck-amd64-linux.so)
==2299==    by 0x109383: hal_ml_create (in /workspace/memory_leak_demo)
==2299==    by 0x109823: buggy_snpe_subplugin::buggy_snpe_subplugin() (in /workspace/memory_leak_demo)
==2299==    by 0x1094D5: main (in /workspace/memory_leak_demo)
==2299== 
==2299== LEAK SUMMARY:
==2299==    definitely lost: 100 bytes in 1 blocks
```

### 🔍 Reading the Valgrind Output

- **`definitely lost: 100 bytes`**: Memory allocated but never freed
- **Stack trace**: Shows exactly where the leak occurred:
  1. `malloc` allocates memory
  2. `hal_ml_create` calls malloc
  3. `buggy_snpe_subplugin` constructor calls hal_ml_create
  4. `main` creates the object

### ✅ The Fix

**Option 1: Manual cleanup before exception**
```cpp
snpe_tizen_hal_subplugin::snpe_tizen_hal_subplugin ()
    : tensor_filter_subplugin (), hal_handle (nullptr)
{
  int ret = hal_ml_create ("snpe", &hal_handle);
  if (ret != HAL_ML_ERROR_NONE) {
    // ✅ FIXED: Clean up before throwing
    if (hal_handle) {
      hal_ml_destroy(hal_handle);
      hal_handle = nullptr;
    }
    
    if (ret == HAL_ML_ERROR_INVALID_PARAMETER) {
      throw std::invalid_argument ("SNPE HAL is not supported");
    }
  }
}
```

**Option 2: RAII wrapper (Best Practice)**
```cpp
class HALHandle {
private:
    hal_ml_h handle;
public:
    HALHandle() : handle(nullptr) {}
    ~HALHandle() { 
        if (handle) hal_ml_destroy(handle); 
    }
    
    int create(const char* backend) {
        return hal_ml_create(backend, &handle);
    }
};

class raii_snpe_subplugin {
private:
    HALHandle hal_handle;  // ✅ Automatic cleanup via destructor
    
public:
    raii_snpe_subplugin() {
        int ret = hal_handle.create("snpe");
        if (ret == HAL_ML_ERROR_INVALID_PARAMETER) {
            // ✅ hal_handle destructor automatically cleans up!
            throw std::invalid_argument("SNPE HAL is not supported");
        }
    }
};
```

### 🧪 Validation Result
```
# Fixed version - Valgrind output:
==2299== HEAP SUMMARY:
==2299==     in use at exit: 0 bytes in 0 blocks
==2299==   total heap usage: 10 allocs, 10 frees, 75,372 bytes allocated
==2299== 
==2299== All heap blocks were freed -- no leaks are possible
```

## Bug Pattern 2: Inconsistent Error Handling

### 🚨 The Problem

Function logs error but continues execution with invalid state.

**Real bug from `tensor_filter_tizen_hal_snpe.cc:186`:**
```cpp
int getModelInfo(...) {
  hal_ml_param_h param = nullptr;
  
  int ret = hal_ml_param_create (&param);
  if (ret != HAL_ML_ERROR_NONE) {
    nns_loge ("Failed to create hal_ml_param");  // ❌ Only logs!
    // Code continues with param == nullptr
  }
  
  // This will crash or cause undefined behavior:
  hal_ml_param_set(param, "key", value);  // ❌ Segfault!
}
```

### 📊 Valgrind Output
```
==3445== Invalid write of size 8
==3445==    at 0x401234: hal_ml_param_set (program.c:123)
==3445==    by 0x405678: getModelInfo (program.c:186)
==3445==  Address 0x0 is not stack'd, malloc'd or (recently) free'd
==3445== 
==3445== Process terminating with default action of signal 11 (SIGSEGV)
```

### 🔍 Reading the Valgrind Output

- **`Invalid write of size 8`**: Writing to invalid memory
- **`Address 0x0`**: Writing to NULL pointer (param == nullptr)
- **`signal 11 (SIGSEGV)`**: Segmentation fault

### ✅ The Fix
```cpp
int getModelInfo(...) {
  hal_ml_param_h param = nullptr;
  
  int ret = hal_ml_param_create (&param);
  if (ret != HAL_ML_ERROR_NONE) {
    nns_loge ("Failed to create hal_ml_param");
    return HAL_ML_ERROR_RUNTIME_ERROR;  // ✅ FIXED: Proper error return
  }
  
  // Safe to use param here
  hal_ml_param_set(param, "key", value);
  
  // Cleanup
  hal_ml_param_destroy(param);
  return HAL_ML_ERROR_NONE;
}
```

## Bug Pattern 3: Double-Free Vulnerabilities

### 🚨 The Problem

Freeing the same memory twice causes heap corruption.

```cpp
void* memory = malloc(100);
free(memory);
// ... some code ...
free(memory);  // ❌ Double-free!
```

### 📊 Valgrind Output
```
==4567== Invalid free() / delete / delete[] / realloc()
==4567==    at 0x484B27F: free (in /usr/libexec/valgrind/vgpreload_memcheck-amd64-linux.so)
==4567==    by 0x401890: main (program.c:45)
==4567==  Address 0x4e54760 is 0 bytes inside a block of size 100 free'd
==4567==    at 0x484B27F: free (in /usr/libexec/valgrind/vgpreload_memcheck-amd64-linux.so)
==4567==    by 0x401856: main (program.c:42)
==4567==  Block was alloc'd at
==4567==    at 0x484A858: malloc (in /usr/libexec/valgrind/vgpreload_memcheck-amd64-linux.so)
==4567==    by 0x401823: main (program.c:40)
```

### 🔍 Reading the Valgrind Output

- **`Invalid free()`**: Attempting to free already-freed memory
- **Stack traces**: Shows both the invalid free and the original free
- **`Block was alloc'd at`**: Shows where memory was originally allocated

### ✅ The Fix
```cpp
void* memory = malloc(100);
free(memory);
memory = nullptr;  // ✅ Prevent double-free

if (memory) {      // ✅ Safe check
    free(memory);
    memory = nullptr;
}
```

## Bug Pattern 4: Use-After-Free

### 🚨 The Problem

Accessing memory after it has been freed.

```cpp
char* buffer = malloc(100);
strcpy(buffer, "Hello");
free(buffer);
printf("%s\n", buffer);  // ❌ Use-after-free!
```

### 📊 Valgrind Output
```
==5678== Invalid read of size 1
==5678==    at 0x484E142: strlen (in /usr/libexec/valgrind/vgpreload_memcheck-amd64-linux.so)
==5678==    by 0x48A0FF4: puts (ioputs.c:35)
==5678==    by 0x401234: main (program.c:45)
==5678==  Address 0x4e54760 is 0 bytes inside a block of size 100 free'd
==5678==    at 0x484B27F: free (in /usr/libexec/valgrind/vgpreload_memcheck-amd64-linux.so)
==5678==    by 0x401220: main (program.c:43)
```

### ✅ The Fix
```cpp
char* buffer = malloc(100);
strcpy(buffer, "Hello");
printf("%s\n", buffer);  // ✅ Use before free
free(buffer);
buffer = nullptr;         // ✅ Prevent use-after-free
```

## Valgrind Command Reference

### Basic Commands

```bash
# Basic leak detection
valgrind --leak-check=full ./program

# Detailed analysis with origin tracking
valgrind --tool=memcheck \
         --leak-check=full \
         --track-origins=yes \
         --show-leak-kinds=all \
         ./program

# With debug symbols (compile with -g)
g++ -g -o program program.cpp
valgrind --leak-check=full ./program

# Using suppression files
valgrind --suppressions=my_suppressions.supp \
         --leak-check=full \
         ./program
```

### Understanding Output

#### Leak Types
1. **Definitely lost**: Memory not reachable - **FIX REQUIRED**
2. **Indirectly lost**: Memory reachable through leaked pointers
3. **Possibly lost**: Memory that might be pointed to - investigate
4. **Still reachable**: Memory pointed to at exit - usually OK

#### Error Types
1. **Invalid read/write**: Using freed or uninitialized memory - **CRITICAL**
2. **Invalid free**: Double-free or freeing invalid pointer - **CRITICAL** 
3. **Mismatched free**: Using wrong deallocator (malloc/delete) - **CRITICAL**
4. **Uninitialized value**: Using uninitialized variables - **WARNING**

## Best Practices

### 1. RAII (Resource Acquisition Is Initialization)
```cpp
class SmartHandle {
    handle_t h;
public:
    SmartHandle() : h(create_handle()) {}
    ~SmartHandle() { destroy_handle(h); }
    handle_t get() const { return h; }
};
```

### 2. Smart Pointers
```cpp
// C++11 compatible:
std::unique_ptr<MyClass> obj(new MyClass());

// C++14 and later (preferred):
// auto obj = std::make_unique<MyClass>();
```

### 3. Exception Safety
```cpp
void function() {
    Resource* r = acquire_resource();
    try {
        // Use resource
        risky_operation();
    } catch (...) {
        release_resource(r);  // ✅ Cleanup in exception path
        throw;
    }
    release_resource(r);
}
```

### 4. Defensive Programming
```cpp
void cleanup(void** ptr) {
    if (ptr && *ptr) {
        free(*ptr);
        *ptr = nullptr;  // ✅ Prevent double-free
    }
}
```

## Integration with CI/CD

### GitHub Actions Example
```yaml
- name: Run Valgrind Tests
  run: |
    valgrind --error-exitcode=1 \
             --leak-check=full \
             --show-leak-kinds=all \
             --track-origins=yes \
             --suppressions=.valgrind_suppressions \
             ./run_tests
```

### Suppression File Example
```
{
   glib_internal_leak
   Memcheck:Leak
   fun:malloc
   obj:*/libglib*
   fun:g_type_register_static
}
```

## Common Pitfalls

### 1. Constructor/Destructor Issues
❌ **Wrong**: Throwing in constructor without cleanup
✅ **Right**: Use RAII or manual cleanup before throw

### 2. Array vs Single Object
❌ **Wrong**: `new[]` with `delete` or `new` with `delete[]`
✅ **Right**: Match `new[]` with `delete[]` and `new` with `delete`

### 3. C/C++ Mixing
❌ **Wrong**: `malloc()` with `delete` or `new` with `free()`
✅ **Right**: Match allocators: `malloc/free`, `new/delete`

### 4. Thread Safety
❌ **Wrong**: Sharing non-thread-safe resources
✅ **Right**: Use mutexes, atomic operations, or thread-local storage

## Running the Demo

```bash
# Compile and run the comprehensive demo
g++ -o memory_leak_demo memory_leak_demo.cpp -std=c++11 -g

# Run normally to see the output
./memory_leak_demo

# Run with Valgrind to see memory errors
valgrind --tool=memcheck \
         --leak-check=full \
         --track-origins=yes \
         --show-leak-kinds=all \
         ./memory_leak_demo
```

The demo will show you:
1. Constructor exception memory leak (buggy vs fixed)
2. RAII-based solution (best practice)
3. Parameter handling errors
4. Double-free scenarios
5. Detailed Valgrind interpretation guide

## Conclusion

Valgrind is an essential tool for C/C++ memory management. The key is to:

1. **Run Valgrind regularly** during development
2. **Fix "definitely lost" leaks immediately**
3. **Use RAII and smart pointers** to prevent leaks
4. **Write exception-safe code** with proper cleanup
5. **Integrate Valgrind into CI/CD** to catch regressions

Remember: **All memory bugs found by Valgrind are real bugs** that can cause crashes, security vulnerabilities, or unpredictable behavior in production.