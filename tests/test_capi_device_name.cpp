#include "parakeet_capi.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

// parakeet_capi_device_name / parakeet_capi_device_description (issue #149 /
// ABI v9): a caller that requested a specific compute device via
// PARAKEET_DEVICE or GGML_VK_VISIBLE_DEVICES has no other way to confirm the
// request was honored rather than silently falling back to CPU.
//
// Env:
//   PARAKEET_TEST_GGUF   model weights (skip 77 if unset) — only the NULL-ctx
//                        case below runs without it.

int main() {
    if (parakeet_capi_abi_version() < 9) {
        std::fprintf(stderr, "test_capi_device_name: abi version < 9\n");
        return 1;
    }

    // NULL ctx must not crash, and both return "" like parakeet_capi_last_error.
    const char* none_name = parakeet_capi_device_name(nullptr);
    const char* none_desc = parakeet_capi_device_description(nullptr);
    if (none_name == nullptr || std::strlen(none_name) != 0 ||
        none_desc == nullptr || std::strlen(none_desc) != 0) {
        std::fprintf(stderr, "test_capi_device_name: NULL ctx did not return \"\"\n");
        return 1;
    }

    const char* gguf = std::getenv("PARAKEET_TEST_GGUF");
    if (!gguf) return 77;  // skip: no model to load

    parakeet_ctx* ctx = parakeet_capi_load(gguf);
    if (!ctx) {
        std::fprintf(stderr, "test_capi_device_name: failed to load %s\n", gguf);
        return 1;
    }

    // Whatever device this build/machine selects, both strings must be
    // non-empty. A CI runner has no GPU, so both are "cpu" there; a GPU
    // build/machine may report a registry ordinal and a physical device
    // description instead — this test doesn't assume which.
    const char* name = parakeet_capi_device_name(ctx);
    const char* desc = parakeet_capi_device_description(ctx);
    if (name == nullptr || std::strlen(name) == 0 ||
        desc == nullptr || std::strlen(desc) == 0) {
        std::fprintf(stderr, "test_capi_device_name: device name/description is empty\n");
        parakeet_capi_free(ctx);
        return 1;
    }
    std::printf("test_capi_device_name: device = %s (%s)\n", name, desc);

    parakeet_capi_free(ctx);
    return 0;
}
