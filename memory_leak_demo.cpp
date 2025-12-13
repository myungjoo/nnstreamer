#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <memory>

// Mock HAL ML functions to simulate the real Tizen HAL library
extern "C" {
    int HAL_ML_ERROR_NONE = 0;
    int HAL_ML_ERROR_INVALID_PARAMETER = -1;
    int HAL_ML_ERROR_RUNTIME_ERROR = -2;
    
    typedef void* hal_ml_h;
    typedef void* hal_ml_param_h;
    
    // Mock implementation that allocates memory (simulates real HAL behavior)
    int hal_ml_create(const char* backend, hal_ml_h* handle) {
        std::cout << "hal_ml_create called for backend: " << backend << std::endl;
        
        // Allocate memory to simulate the real HAL library
        *handle = malloc(100);
        std::cout << "  -> Allocated memory at: " << *handle << std::endl;
        
        // Simulate different error conditions based on backend name
        if (strcmp(backend, "fail_invalid") == 0) {
            return HAL_ML_ERROR_INVALID_PARAMETER;
        } else if (strcmp(backend, "fail_runtime") == 0) {
            return HAL_ML_ERROR_RUNTIME_ERROR;
        }
        
        return HAL_ML_ERROR_NONE;
    }
    
    void hal_ml_destroy(hal_ml_h handle) {
        std::cout << "hal_ml_destroy called for handle: " << handle << std::endl;
        if (handle) {
            free(handle);
            std::cout << "  -> Memory freed successfully" << std::endl;
        }
    }
    
    int hal_ml_param_create(hal_ml_param_h* param) {
        *param = malloc(50);  // Smaller allocation for param
        std::cout << "hal_ml_param_create allocated: " << *param << std::endl;
        return HAL_ML_ERROR_NONE;
    }
    
    void hal_ml_param_destroy(hal_ml_param_h param) {
        std::cout << "hal_ml_param_destroy called for: " << param << std::endl;
        if (param) {
            free(param);
        }
    }
}

// Simplified tensor_filter_subplugin base class
class tensor_filter_subplugin {
public:
    tensor_filter_subplugin() = default;
    virtual ~tensor_filter_subplugin() = default;
};

// =============================================================================
// DEMONSTRATION 1: Constructor Exception Memory Leak (ORIGINAL BUG PATTERN)
// =============================================================================
// This reproduces the exact issue found in tensor_filter_tizen_hal_snpe.cc:73
// and tensor_filter_tizen_hal_vivante.cc:71

class buggy_snpe_subplugin : public tensor_filter_subplugin {
private:
    hal_ml_h hal_handle;
    
public:
    buggy_snpe_subplugin() : tensor_filter_subplugin(), hal_handle(nullptr) {
        std::cout << "\n--- BUGGY Constructor: Starting ---" << std::endl;
        
        int ret = hal_ml_create("fail_invalid", &hal_handle);
        if (ret == HAL_ML_ERROR_INVALID_PARAMETER) {
            // BUG: hal_handle contains allocated memory but we don't clean it up!
            // When constructor throws, destructor is NEVER called
            std::cout << "ERROR: About to throw exception WITHOUT cleanup!" << std::endl;
            throw std::invalid_argument("SNPE HAL is not supported");
        }
        if (ret != HAL_ML_ERROR_NONE) {
            // Another path with the same bug
            throw std::runtime_error("Failed to initialize SNPE HAL ML");
        }
        
        std::cout << "--- BUGGY Constructor: Completed successfully ---" << std::endl;
    }
    
    ~buggy_snpe_subplugin() {
        std::cout << "--- BUGGY Destructor: Cleaning up ---" << std::endl;
        if (hal_handle) {
            hal_ml_destroy(hal_handle);
            hal_handle = nullptr;
        }
        std::cout << "--- BUGGY Destructor: Done ---" << std::endl;
    }
};

// =============================================================================
// DEMONSTRATION 2: Fixed Constructor with Proper Exception Safety
// =============================================================================

class fixed_snpe_subplugin : public tensor_filter_subplugin {
private:
    hal_ml_h hal_handle;
    
public:
    fixed_snpe_subplugin() : tensor_filter_subplugin(), hal_handle(nullptr) {
        std::cout << "\n--- FIXED Constructor: Starting ---" << std::endl;
        
        int ret = hal_ml_create("fail_invalid", &hal_handle);
        if (ret != HAL_ML_ERROR_NONE) {
            // FIXED: Clean up BEFORE throwing exception
            std::cout << "ERROR detected, cleaning up before exception..." << std::endl;
            if (hal_handle) {
                hal_ml_destroy(hal_handle);
                hal_handle = nullptr;
            }
            
            if (ret == HAL_ML_ERROR_INVALID_PARAMETER) {
                throw std::invalid_argument("SNPE HAL is not supported");
            } else {
                throw std::runtime_error("Failed to initialize SNPE HAL ML");
            }
        }
        
        std::cout << "--- FIXED Constructor: Completed successfully ---" << std::endl;
    }
    
    ~fixed_snpe_subplugin() {
        std::cout << "--- FIXED Destructor: Cleaning up ---" << std::endl;
        if (hal_handle) {
            hal_ml_destroy(hal_handle);
            hal_handle = nullptr;
        }
        std::cout << "--- FIXED Destructor: Done ---" << std::endl;
    }
};

// =============================================================================
// DEMONSTRATION 3: RAII-based Solution (Best Practice)
// =============================================================================

class raii_snpe_subplugin : public tensor_filter_subplugin {
private:
    // RAII wrapper for hal_ml_h
    class HALHandle {
    private:
        hal_ml_h handle;
    public:
        HALHandle() : handle(nullptr) {}
        
        ~HALHandle() {
            if (handle) {
                hal_ml_destroy(handle);
            }
        }
        
        int create(const char* backend) {
            return hal_ml_create(backend, &handle);
        }
        
        hal_ml_h get() const { return handle; }
        
        // Prevent copying
        HALHandle(const HALHandle&) = delete;
        HALHandle& operator=(const HALHandle&) = delete;
    };
    
    HALHandle hal_handle;  // Automatic cleanup via destructor
    
public:
    raii_snpe_subplugin() : tensor_filter_subplugin() {
        std::cout << "\n--- RAII Constructor: Starting ---" << std::endl;
        
        int ret = hal_handle.create("fail_invalid");
        if (ret == HAL_ML_ERROR_INVALID_PARAMETER) {
            // RAII: hal_handle destructor will automatically clean up!
            throw std::invalid_argument("SNPE HAL is not supported");
        }
        if (ret != HAL_ML_ERROR_NONE) {
            throw std::runtime_error("Failed to initialize SNPE HAL ML");
        }
        
        std::cout << "--- RAII Constructor: Completed successfully ---" << std::endl;
    }
    
    ~raii_snpe_subplugin() {
        std::cout << "--- RAII Destructor: HALHandle will auto-cleanup ---" << std::endl;
    }
};

// =============================================================================
// DEMONSTRATION 4: Parameter Creation Error Handling Bug
// =============================================================================

class buggy_param_handling {
public:
    int getModelInfo_buggy() {
        hal_ml_param_h param = nullptr;
        
        std::cout << "\n--- BUGGY Parameter Handling ---" << std::endl;
        
        // Simulate param creation failure
        int ret = -1; // Force failure
        if (ret != HAL_ML_ERROR_NONE) {
            std::cout << "ERROR: Failed to create hal_ml_param" << std::endl;
            // BUG: Only logs error, doesn't return! 
            // Code continues with param == nullptr
        }
        
        // This will likely crash or cause undefined behavior
        std::cout << "About to use potentially NULL param: " << param << std::endl;
        // In real code: hal_ml_param_set(param, "key", value); // CRASH!
        
        return 0;
    }
    
    int getModelInfo_fixed() {
        hal_ml_param_h param = nullptr;
        
        std::cout << "\n--- FIXED Parameter Handling ---" << std::endl;
        
        int ret = hal_ml_param_create(&param);
        if (ret != HAL_ML_ERROR_NONE) {
            std::cout << "ERROR: Failed to create hal_ml_param" << std::endl;
            return HAL_ML_ERROR_RUNTIME_ERROR;  // FIXED: Proper error return
        }
        
        std::cout << "SUCCESS: Created param: " << param << std::endl;
        
        // Use param safely...
        
        // Cleanup
        hal_ml_param_destroy(param);
        return HAL_ML_ERROR_NONE;
    }
};

// =============================================================================
// DEMONSTRATION 5: Double-Free Scenario
// =============================================================================

class double_free_demo {
public:
    static void demonstrate_double_free() {
        std::cout << "\n--- DOUBLE-FREE DEMONSTRATION ---" << std::endl;
        
        // Simulate the object lifecycle issue
        void* memory = malloc(200);
        std::cout << "Allocated memory: " << memory << std::endl;
        
        std::cout << "First free()..." << std::endl;
        free(memory);
        
        std::cout << "Attempting second free() - THIS WILL CAUSE ISSUES!" << std::endl;
        // UNCOMMENT NEXT LINE TO SEE DOUBLE-FREE ERROR:
        // free(memory);  // ❌ Double-free error!
        
        std::cout << "Double-free demonstration skipped to avoid crash" << std::endl;
    }
};

// =============================================================================
// MAIN FUNCTION WITH COMPREHENSIVE TESTING
// =============================================================================

void print_separator(const char* title) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "   " << title << std::endl;
    std::cout << std::string(60, '=') << std::endl;
}

int main() {
    std::cout << "Memory Bug Demonstration Program" << std::endl;
    std::cout << "Run with: valgrind --tool=memcheck --leak-check=full --track-origins=yes ./memory_leak_demo" << std::endl;
    
    // Test 1: Constructor Exception Memory Leak (Original Bug)
    print_separator("TEST 1: Constructor Exception Memory Leak (BUGGY)");
    try {
        buggy_snpe_subplugin* obj = new buggy_snpe_subplugin();
        delete obj; // Won't reach here due to exception
    } catch (const std::exception& e) {
        std::cout << "❌ Caught exception: " << e.what() << std::endl;
        std::cout << "❌ MEMORY LEAKED: hal_handle memory not freed!" << std::endl;
        std::cout << "❌ Valgrind will report: '100 bytes in 1 blocks are definitely lost'" << std::endl;
    }
    
    // Test 2: Fixed Constructor (No Memory Leak)
    print_separator("TEST 2: Fixed Constructor Exception Handling");
    try {
        fixed_snpe_subplugin* obj = new fixed_snpe_subplugin();
        delete obj; // Won't reach here due to exception
    } catch (const std::exception& e) {
        std::cout << "✅ Caught exception: " << e.what() << std::endl;
        std::cout << "✅ NO MEMORY LEAK: Cleanup performed before exception!" << std::endl;
        std::cout << "✅ Valgrind will report: '0 bytes lost'" << std::endl;
    }
    
    // Test 3: RAII-based Solution (Best Practice)
    print_separator("TEST 3: RAII-based Solution (BEST PRACTICE)");
    try {
        raii_snpe_subplugin* obj = new raii_snpe_subplugin();
        delete obj; // Won't reach here due to exception
    } catch (const std::exception& e) {
        std::cout << "✅ Caught exception: " << e.what() << std::endl;
        std::cout << "✅ RAII CLEANUP: Automatic resource management!" << std::endl;
        std::cout << "✅ Valgrind will report: '0 bytes lost'" << std::endl;
    }
    
    // Test 4: Parameter Handling
    print_separator("TEST 4: Parameter Creation Error Handling");
    buggy_param_handling param_demo;
    
    std::cout << "\nTesting buggy parameter handling:" << std::endl;
    param_demo.getModelInfo_buggy();
    
    std::cout << "\nTesting fixed parameter handling:" << std::endl;
    param_demo.getModelInfo_fixed();
    
    // Test 5: Double-free demonstration
    print_separator("TEST 5: Double-Free Scenario");
    double_free_demo::demonstrate_double_free();
    
    // Summary
    print_separator("VALGRIND ANALYSIS GUIDE");
    std::cout << R"(
🔍 HOW TO READ VALGRIND OUTPUT:

1. DEFINITELY LOST: Memory allocated but never freed
   Example: "100 bytes in 1 blocks are definitely lost"
   → FIX: Add proper cleanup before exceptions

2. INDIRECTLY LOST: Memory reachable through leaked pointers
   → Usually fixed by fixing definitely lost blocks

3. POSSIBLY LOST: Memory that might be pointed to
   → Often false positives, but investigate

4. STILL REACHABLE: Memory pointed to at exit
   → Usually not a problem for long-running programs

5. INVALID READ/WRITE: Using freed or uninitialized memory
   → Critical bugs that can cause crashes

🛠️  COMMON MEMORY BUG PATTERNS:

1. Constructor Exception Leaks:
   ❌ allocate() → exception thrown → destructor not called
   ✅ allocate() → cleanup → exception thrown

2. Inconsistent Error Handling:
   ❌ if (error) { log_error(); /* continue anyway */ }
   ✅ if (error) { log_error(); return ERROR_CODE; }

3. Double-free:
   ❌ free(ptr); /* ... */ free(ptr);
   ✅ free(ptr); ptr = nullptr;

4. Use-after-free:
   ❌ free(ptr); /* ... */ *ptr = value;
   ✅ free(ptr); ptr = nullptr; /* can't dereference */

5. RAII (Resource Acquisition Is Initialization):
   ✅ Use smart pointers, destructors, and RAII classes
   ✅ Automatic cleanup on scope exit

🧪 TESTING COMMANDS:

Basic leak check:
  valgrind --leak-check=full ./program

Detailed analysis:
  valgrind --tool=memcheck --leak-check=full --track-origins=yes --show-leak-kinds=all ./program

With suppressions:
  valgrind --suppressions=valgrind_suppression --leak-check=full ./program
)" << std::endl;
    
    return 0;
}