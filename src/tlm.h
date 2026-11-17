// tlm as a library: link src/libtlm.c (which compiles src/tlm.c unchanged). One instance per process: the model and
// the KV cache are global, and tlm_load pins the calling thread to a core, so load and generate on the same thread.
#ifndef TLM_H
#define TLM_H

enum { TLM_BOS = 128000, TLM_HDR_START = 128006, TLM_HDR_END = 128007 };  // Llama 3 special token ids
enum { TLM_EOM = 128008, TLM_EOT = 128009 };

int tlm_load(const char *path);  // 0 on success
int tlm_context(void);           // size of the ctx arrays passed to tlm_generate (max context)
int tlm_prompt_cap(void);        // longest prompt tlm_generate accepts
int tlm_threads(void);           // threads in use (TLM_THREADS or one per physical core)
int tlm_has_draft(void);         // 1 if the model file has a draft head

// Text -> token ids (no special tokens): returns the count, or -1 if more than cap.
int tlm_encode(const char *text, int *out, int cap);

// The Llama 3.2 chat template for one user message (system header with today's date): returns the token count, or
// -1 if too long. Multi-turn and other templates: build ctx from tlm_encode and the header/end ids yourself.
int tlm_chat_prompt(const char *user, int *ctx);

// Greedy generation after ctx[0..n): appends the reply to ctx (stopping after an end token, <|eot_id|>, <|eom_id|>
// or <|end_of_text|>, after max_new tokens or at the context limit) and returns the new length m, or -1 if n is too
// long. emit is called for every new token except special ones (those stay in ctx but are not emitted).
// k: tokens drafted per pass (0 = none, 3 = default). from: the KV cache already holds ctx[0..from) from earlier
// calls on the same tokens; after a call returning m it holds ctx[0..m - 1), so a conversation that keeps ctx and
// appends to it passes the previous m - 1 and only the new tokens are processed. The output is the same as with
// from = 0.
int tlm_generate(int *ctx, int n, int from, int max_new, int k, void (*emit)(int token), int *passes);

int tlm_token(int token, const char **bytes);  // the token's bytes (UTF-8 fragments for multi-byte characters)

#endif
