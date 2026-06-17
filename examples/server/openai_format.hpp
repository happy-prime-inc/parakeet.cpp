#pragma once
#include "transcription.hpp"   // pk::Transcription, pk::Word
#include <string>

namespace pkserver {

enum class Format { kJson, kText, kVerboseJson };

// Parse an OpenAI response_format value. Empty string maps to kJson (the
// OpenAI default). Returns false for an unrecognized value.
bool parse_format(const std::string& s, Format& out);

struct Response {
    std::string body;
    std::string content_type;
};

// Build the response body and content type for a finished transcription.
// duration_sec is the decoded audio length in seconds. include_words controls
// whether the verbose_json "words" array is emitted (OpenAI gates it on
// timestamp_granularities[] containing "word").
Response format_transcription(const pk::Transcription& tr, Format fmt,
                              double duration_sec, bool include_words);

// JSON-escape a UTF-8 string (quote, backslash, control chars). Exposed for the
// error envelope and tests.
std::string json_escape(const std::string& s);

// Build an OpenAI-style error envelope: {"error":{"message":..,"type":..}}.
std::string error_body(const std::string& message, const std::string& type);

} // namespace pkserver
