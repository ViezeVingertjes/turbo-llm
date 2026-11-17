// tlm as a library (API: tlm.h). src/tlm.c is compiled unchanged, with its main renamed.
#define main tlm_main
#include "tlm.c"
#undef main
#include "tlm.h"

int tlm_load(const char *path) { return load(path); }
int tlm_context(void) { return T; }
int tlm_prompt_cap(void) { return PCAP; }
int tlm_threads(void) { return NT; }
int tlm_has_draft(void) { return has_draft; }
int tlm_encode(const char *text, int *out, int cap) { return encode(text, out, cap); }
int tlm_chat_prompt(const char *user, int *ctx) { return chat_prompt(user, ctx); }

int tlm_generate(int *ctx, int n, int from, int max_new, int k, void (*emit)(int token), int *passes) {
  if (n < 1 || n > PCAP) return -1;  // the CLI checks this in chat_prompt
  return generate(ctx, n, from, max_new, k, emit, passes);
}

int tlm_token(int token, const char **bytes) {
  *bytes = (const char *)vbytes + voff[token];
  return (int)(voff[token + 1] - voff[token]);
}
