#include "model_fetch.hpp"

#include <cstdio>
#include <string>

static int fails = 0;
static void check(bool ok, const char* msg) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", msg); ++fails; }
}

int main() {
    using namespace pkserver;
    ModelSource s;
    std::string err;

    // Existing file resolves to a local path (run from project root).
    check(resolve_model("tests/fixtures/speech.wav", s, err), "wav resolves");
    check(s.kind == ModelSource::kLocalPath, "wav is local path");
    check(s.value == "tests/fixtures/speech.wav", "wav path preserved");

    // Explicit URL.
    check(resolve_model("https://example.com/dir/foo.gguf", s, err), "url resolves");
    check(s.kind == ModelSource::kUrl, "url is kUrl");
    check(s.value == "https://example.com/dir/foo.gguf", "url preserved");
    check(s.cache_name == "foo.gguf", "url basename");

    // Known alias maps to the collective repo.
    check(resolve_model("tdt_ctc-110m-q4_k", s, err), "alias resolves");
    check(s.kind == ModelSource::kUrl, "alias is kUrl");
    check(s.value ==
          "https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/"
          "tdt_ctc-110m-q4_k.gguf", "alias url");
    check(s.cache_name == "tdt_ctc-110m-q4_k.gguf", "alias cache name");

    // Bare <name>.gguf goes to the collective repo verbatim.
    check(resolve_model("custom-thing.gguf", s, err), "bare gguf resolves");
    check(s.value ==
          "https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/"
          "custom-thing.gguf", "bare gguf url");

    // Unknown bare name is an error listing aliases.
    check(!resolve_model("definitely-not-a-model", s, err), "unknown rejected");
    check(!err.empty(), "unknown sets err");

    // A URL whose basename carries shell metacharacters is rejected by the
    // safe_name guard (the security control on the download path).
    check(!resolve_model("https://h/foo;rm.gguf", s, err), "unsafe ; rejected");
    check(!resolve_model("https://h/foo'bar.gguf", s, err), "unsafe quote rejected");
    check(!resolve_model("https://h/..", s, err), "dotdot basename rejected");

    if (fails) { std::fprintf(stderr, "%d checks failed\n", fails); return 1; }
    std::printf("test_server_resolve: OK\n");
    return 0;
}
