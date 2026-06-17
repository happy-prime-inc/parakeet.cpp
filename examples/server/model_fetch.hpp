#pragma once
#include <string>
#include <utility>
#include <vector>

namespace pkserver {

// Classified result of a --model argument.
struct ModelSource {
    enum Kind { kLocalPath, kUrl };
    Kind kind = kLocalPath;
    std::string value;       // local path (kLocalPath) or download URL (kUrl)
    std::string cache_name;  // for kUrl: filename to cache as (e.g. foo.gguf)
};

// The collective repo every alias points at.
extern const char* kCollectionRepo;  // "mudler/parakeet-cpp-gguf"

// Alias table: {alias, collective-repo filename}. Exposed for help text/tests.
const std::vector<std::pair<std::string, std::string>>& model_aliases();

// Classify a --model argument WITHOUT touching the network:
//   - existing file path -> kLocalPath
//   - http(s):// URL      -> kUrl (cache_name is the URL basename)
//   - known alias         -> kUrl (collective-repo URL)
//   - bare <name>.gguf    -> kUrl (collective-repo URL)
//   - anything else       -> false; err lists known aliases
bool resolve_model(const std::string& arg, ModelSource& out, std::string& err);

// The default cache directory: $PARAKEET_CACHE_DIR, else
// ${XDG_CACHE_HOME:-$HOME/.cache}/parakeet.cpp/models.
std::string default_cache_dir();

// Ensure a ModelSource is on disk and return the local path.
//   kLocalPath: returns value unchanged.
//   kUrl: downloads to <cache_dir>/<cache_name> via curl (skips if present).
// Throws std::runtime_error on failure (download error, no curl/wget, etc.).
std::string fetch_model(const ModelSource& src, const std::string& cache_dir);

} // namespace pkserver
