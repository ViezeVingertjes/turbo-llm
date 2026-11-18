// tlm: Llama 3.2 1B Instruct on AVX2. Exact bf16 weights streamed from a 12-bit lossless format, fp32 math.
// usage: tlm model.tlm "prompt" [max_new_tokens=256] [max_draft=3]
#define _GNU_SOURCE
#include <fcntl.h>
#include <immintrin.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

// Model dimensions. T: max context; MB: max tokens per draft-verification pass; PB: tokens per prompt chunk (64: the
// fastest of 32/64/128 for the prompt kernel, tools/gemmbench.c).
enum { D = 2048, NL = 16, NH = 32, HD = 64, KV = 512, QKV = D + 2 * KV, F = 8192, V = 128256, T = 8192, MB = 8 };
enum { PB = 64 };
enum { MAXT = 64 };           // max threads
enum { PCAP = T - 2 * MB };   // max prompt tokens: leaves room for drafts and the final tokens
// Special tokens: ids >= SPECIAL are never printed. Generation ends at the model's end tokens (generation_config's
// eos_token_id: <|end_of_text|>, <|eom_id|>, <|eot_id|>); other special tokens are part of the reply.
enum { SPECIAL = 128000, BOS = 128000, END_OF_TEXT = 128001, HDR_START = 128006, HDR_END = 128007, EOM = 128008 };
enum { EOT = 128009 };

// ---- Model file: every item starts on an ALIGN-byte boundary --------------------------------------------------------

enum { ALIGN = 64 };
static const char MAGIC[] = "TLM1";  // model file format version
static const uint8_t *P, *PEND;  // parse position, end of file

static const void *take(size_t n) {
  const void *r = P;
  P += (n + ALIGN - 1) & ~(size_t)(ALIGN - 1);
  if (P > PEND) {
    fprintf(stderr, "model file truncated or wrong version\n");
    exit(1);
  }
  return r;
}

// Weight matrices. 12-bit format: per block of BW weights, BW/2 bytes of 4-bit codes for the bf16 high byte (sign +
// 7 exponent bits, decoded through a 16-entry per-matrix table) and BW raw low bytes. Code 15 is an escape: the weight
// decodes as 0 and its true value is in a sorted exception list (flat index ei, value ev). Layer matrices are stored
// in tiles of TR rows x TC columns (per TR-row block: its column tiles in order, each TR row segments), so a
// matrix-times-tokens pass reuses a small activation tile; the embedding matrix stays row-major (read by rows).
// q5 matrices are int5 (see coarse()), used for the output-head screening copy and the draft head.
enum { BW = 64, B12 = BW * 3 / 2, TR = 8, TC = 512 };
_Static_assert(QKV % TR == 0 && D % TR == 0 && 2 * F % TR == 0 && D % TC == 0 && F % TC == 0 && TC % BW == 0,
               "layer matrices must split into whole TR x TC tiles");
static inline size_t row12(uint32_t cols) { return (size_t)cols * 3 / 2; }   // bytes per 12-bit row
static inline size_t row5(uint32_t cols) { return (size_t)cols * 21 / 32; }  // bytes per int5 row
typedef struct {
  const uint8_t *w;
  const uint32_t *ei;
  const float *ev;
  uint32_t ne, rows, cols, q5;
  __m256i lut;
} Mat;

static Mat mat(uint32_t rows, uint32_t cols) {
  Mat m = {.rows = rows, .cols = cols};
  m.lut = _mm256_broadcastsi128_si256(_mm_load_si128(take(16)));
  m.ne = *(const uint32_t *)take(4);
  m.ei = take(4 * m.ne);
  m.ev = take(4 * m.ne);
  m.w = take(rows * row12(cols));
  return m;
}

static Mat mat5(uint32_t rows, uint32_t cols) {
  Mat m = {.rows = rows, .cols = cols, .q5 = 1};
  m.w = take(rows * row5(cols));
  return m;
}

static uint32_t first_exc(const Mat *m, uint32_t idx) {  // index of the first exception at or after flat index idx
  uint32_t lo = 0, hi = m->ne;
  while (lo < hi) {
    uint32_t md = (lo + hi) / 2;
    if (m->ei[md] < idx) lo = md + 1;
    else hi = md;
  }
  return lo;
}

// ---- Weight decoding and dot-product kernels ------------------------------------------------------------------------

// (low, high) byte vectors -> 4 fp32 vectors: in-lane unpack to bf16 words, then each word into the top half of a dword
static inline __attribute__((always_inline)) void widen(__m256i l, __m256i h, __m256 *w) {
  const __m256i hm = _mm256_set1_epi32(0xffff0000);
  __m256i u = _mm256_unpacklo_epi8(l, h), v = _mm256_unpackhi_epi8(l, h);
  w[0] = _mm256_castsi256_ps(_mm256_slli_epi32(u, 16));
  w[1] = _mm256_castsi256_ps(_mm256_and_si256(u, hm));
  w[2] = _mm256_castsi256_ps(_mm256_slli_epi32(v, 16));
  w[3] = _mm256_castsi256_ps(_mm256_and_si256(v, hm));
}

// one 12-bit block (B12 bytes) -> BW fp32 weights in column order (the converter pre-permuted the slots)
static inline __attribute__((always_inline)) void dec(const uint8_t *p, __m256i lut, __m256 w[8]) {
  const __m256i codes = _mm256_load_si256((const void *)p), nibble = _mm256_set1_epi8(15);
  __m256i lo = _mm256_and_si256(codes, nibble), hi = _mm256_and_si256(_mm256_srli_epi16(codes, 4), nibble);
  widen(_mm256_load_si256((const void *)(p + 32)), _mm256_shuffle_epi8(lut, lo), w);
  widen(_mm256_load_si256((const void *)(p + 64)), _mm256_shuffle_epi8(lut, hi), w + 4);
}

static inline float hsum(__m256 v) {
  __m128 x = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  x = _mm_add_ps(x, _mm_movehl_ps(x, x));
  return _mm_cvtss_f32(_mm_add_ss(x, _mm_movehdup_ps(x)));
}

// 12-bit row segment (bytes p..e) times G <= 4 tokens (x_b = x + b * C) sharing each decoded block, accumulated into
// a[b]. Every token uses the same 2 accumulators in the same order regardless of G, batch size or tiling, so batched
// (verification) results are bitwise identical to unbatched ones.
static inline __attribute__((always_inline)) void part(const uint8_t *p, const uint8_t *e, __m256i lut,
                                                       const float *x, uint32_t C, __m256 a[][2], int G) {
  __m256 w[8];
  for (; p < e; p += B12, x += BW) {
    dec(p, lut, w);
    for (int b = 0; b < G; b++)
      for (int k = 0; k < 8; k++) a[b][k & 1] = _mm256_fmadd_ps(w[k], _mm256_loadu_ps(x + b * C + 8 * k), a[b][k & 1]);
  }
}

// A whole 12-bit row (bytes p..e) times G <= 4 tokens: y[b * R] = result.
static inline __attribute__((always_inline)) void rows(const uint8_t *p, const uint8_t *e, __m256i lut,
                                                       const float *x, uint32_t C, float *y, uint32_t R, int G) {
  __m256 a[4][2] = {0};
  part(p, e, lut, x, C, a, G);
  for (int b = 0; b < G; b++) y[b * R] = hsum(_mm256_add_ps(a[b][0], a[b][1]));
}

// int5 rows: sign-magnitude, one fp16 scale per BW weights. A row of C weights is C/2 bytes of 4-bit magnitudes (slot
// order as dec), C/8 bytes of sign bits and C/32 bytes of scales. Magnitude k becomes the fp32 k through two byte
// tables (the high bytes of float(k)) and the sign bit is OR-ed into the top byte. out[b * ostride] = w_hat . x_b for
// G <= 4 tokens (x_b = x + b * C) sharing the decode. Used by the draft head; the output head's screen is iscreen().
enum { HQ = D * 21 / 32 };  // bytes per int5 row of D weights
static inline __attribute__((always_inline)) void coarse(const uint8_t *p, uint32_t C, const float *x, float *out,
                                                         uint32_t ostride, int G) {
  const __m256i nibble = _mm256_set1_epi8(15), top = _mm256_set1_epi8((char)0x80),
                bit = _mm256_set1_epi64x(0x8040201008040201),  // byte i of each 8: 1 << i
                spread = _mm256_setr_epi64x(0, 0x0101010101010101, 0x0202020202020202, 0x0303030303030303),
                // byte 3 and byte 2 of float(k) for k = 0..15
                lut3 = _mm256_broadcastsi128_si256(_mm_setr_epi8(0, 63, 64, 64, 64, 64, 64, 64,  //
                                                                 65, 65, 65, 65, 65, 65, 65, 65)),
                lut2 = _mm256_broadcastsi128_si256(_mm_setr_epi8(0, -128, 0, 64, -128, -96, -64, -32,  //
                                                                 0, 16, 32, 48, 64, 80, 96, 112));
  const uint32_t *sgn = (const uint32_t *)(p + C / 2);
  const uint16_t *sc = (const uint16_t *)(p + C / 2 + C / 8);
  __m256 tot[4] = {0}, w[4];
  for (uint32_t g = 0; g < C / BW; g++, x += BW) {
    const __m256i mag = _mm256_load_si256((const void *)(p + 32 * g)),
                  k[2] = {_mm256_and_si256(mag, nibble), _mm256_and_si256(_mm256_srli_epi16(mag, 4), nibble)};
    __m256 in[4][2] = {0};  // two independent half-group chains per token: half the FMA latency
    for (int i = 0; i < 2; i++) {
      __m256i bits = _mm256_shuffle_epi8(_mm256_set1_epi32(sgn[2 * g + i]), spread);  // sign bit j -> byte j
      __m256i neg = _mm256_cmpeq_epi8(_mm256_and_si256(bits, bit), bit);
      __m256i hi = _mm256_or_si256(_mm256_shuffle_epi8(lut3, k[i]), _mm256_and_si256(neg, top));
      widen(_mm256_shuffle_epi8(lut2, k[i]), hi, w);
      for (int b = 0; b < G; b++)
        for (int j = 0; j < 4; j++)
          in[b][i] = _mm256_fmadd_ps(w[j], _mm256_loadu_ps(x + b * C + 32 * i + 8 * j), in[b][i]);
    }
    const __m256 s = _mm256_set1_ps(_cvtsh_ss(sc[g]));
    for (int b = 0; b < G; b++) tot[b] = _mm256_fmadd_ps(_mm256_add_ps(in[b][0], in[b][1]), s, tot[b]);
  }
  for (int b = 0; b < G; b++) out[b * ostride] = hsum(tot[b]);
}

// Prompt kernel: RB rows of decoded weights (w, row stride TC) times VB groups of 8 tokens (packed x: group v, column
// c at x + v * vs + 8 * c) over SC columns, added to s[i * PB / 8 + v]. Lanes are tokens, weights are broadcast: 12
// accumulators + 3 x vectors + 1 weight = the 16 ymm registers, and 7 loads per 12 FMAs (the decode kernel needs 2
// accumulators per (row, token), so it reuses each load only 2-3 times). Each (row, token) sums its SC columns in one
// FMA chain from 0, then adds that to its total: the order is fixed by the columns alone, so a prompt token's result
// depends neither on the other tokens of its chunk nor on how the prompt is split. It differs from the decode kernel's
// by rounding (outputs stay token-identical, make verify); chains of SC keep the error at the decode kernel's level
// (one chain over all columns had 3-4x its logit error against the fp32 reference).
// SC = 128: a sub-block's packed x (VB * 8 tokens x SC columns, 12 KB) stays in L1 with its 4 KB of weights.
enum { RB = 4, VB = 3, SC = 128 };
static inline __attribute__((always_inline)) void pk(const float *w, const float *x, size_t vs, __m256 *s, int V) {
  __m256 a[RB][VB];
  for (int i = 0; i < RB; i++)
    for (int v = 0; v < V; v++) a[i][v] = _mm256_setzero_ps();
  for (int c = 0; c < SC; c++) {
    __m256 xv[VB];
    for (int v = 0; v < V; v++) xv[v] = _mm256_load_ps(x + v * vs + 8 * c);
    for (int i = 0; i < RB; i++) {
      const __m256 wb = _mm256_broadcast_ss(w + i * TC + c);
      for (int v = 0; v < V; v++) a[i][v] = _mm256_fmadd_ps(wb, xv[v], a[i][v]);
    }
  }
  for (int i = 0; i < RB; i++)
    for (int v = 0; v < V; v++) s[i * PB / 8 + v] = _mm256_add_ps(s[i * PB / 8 + v], a[i][v]);
}

// ---- Spinning thread pool: run(f, a) calls f(t, a) on threads t = 0..NT-1 (0 = the caller) and waits ----------------

static int NT, NCORE, CORE_CPU[MAXT];  // threads in use; physical cores (allowed CPUs) and the first CPU of each
static void (*job)(int, void *);
static void *jarg;
static atomic_int gen, done;

// the physical cores in the allowed CPU set, each with its first allowed CPU
static void scan_cores(void) {
  cpu_set_t ok;
  char path[96];
  int core[MAXT][2], id[2];  // (core, package) of each core found
  if (sched_getaffinity(0, sizeof ok, &ok)) {  // unknown: assume one core
    CPU_ZERO(&ok);
    CPU_SET(0, &ok);
  }
  for (int c = 0; c < CPU_SETSIZE && NCORE < MAXT; c++) {
    if (!CPU_ISSET(c, &ok)) continue;
    for (int k = 0; k < 2; k++) {
      const char *what = k ? "physical_package_id" : "core_id";
      snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/%s", c, what);
      FILE *f = fopen(path, "r");
      if (!f || fscanf(f, "%d", id + k) != 1) id[k] = c;  // no topology: every CPU is its own core
      if (f) fclose(f);
    }
    int seen = 0;
    for (int j = 0; j < NCORE; j++) seen |= core[j][0] == id[0] && core[j][1] == id[1];
    if (seen) continue;
    core[NCORE][0] = id[0], core[NCORE][1] = id[1];
    CORE_CPU[NCORE++] = c;
  }
}

static void pin(int t) {  // thread t -> its own physical core (more threads than cores: the rest stay unpinned)
  if (t >= NCORE) return;
  cpu_set_t one;
  CPU_ZERO(&one);
  CPU_SET(CORE_CPU[t], &one);
  pthread_setaffinity_np(pthread_self(), sizeof one, &one);
}

static void *worker(void *a) {
  int t = (int)(intptr_t)a;
  pin(t);
  for (int g = 0;;) {
    while (atomic_load_explicit(&gen, memory_order_acquire) == g) _mm_pause();
    g++;
    job(t, jarg);
    atomic_fetch_add_explicit(&done, 1, memory_order_release);
  }
  return 0;
}

static void run(void (*f)(int, void *), void *a) {
  job = f, jarg = a;
  atomic_store_explicit(&done, 0, memory_order_relaxed);
  atomic_fetch_add_explicit(&gen, 1, memory_order_release);
  f(0, a);
  while (atomic_load_explicit(&done, memory_order_acquire) < NT - 1) _mm_pause();
}

// ---- Matrix times tokens ------------------------------------------------------------------------------------------

static float silu(float x) { return x / (1.0f + expf(-x)); }

// tokens b..b+G-1 in groups of <= 4 with a compile-time G per call, so every kernel is fully specialized
#define G_SWITCH(CALL) \
  switch (nb - b) { case 1: CALL(1); break; case 2: CALL(2); break; case 3: CALL(3); break; default: CALL(4); }

// Prompt chunks' input, packed for pk(): PX[(v * C + c) * 8 + l] = x[(8v + l) * C + c], zero past the nb tokens.
static _Alignas(64) float PX[PB * F];

// y[b * rows + r] = W x_b. With glu: glu[b * rows/2 + i] = silu(y[2i]) * y[2i + 1] (gate/up rows interleaved).
// Each thread owns a contiguous range of rows (whole TR-row blocks, keeping gate/up pairs together). prompt: 12-bit
// matrices use the prompt kernel (pk) on PX.
typedef struct { const Mat *m; const float *x; float *y; int nb; float *glu; int prompt; } MM;

static void pack_job(int t, void *arg) {  // x -> PX, each thread its own columns, 8 at a time
  const MM *a = arg;
  uint32_t C = a->m->cols, c0 = (uint64_t)C / 8 * t / NT * 8, c1 = (uint64_t)C / 8 * (t + 1) / NT * 8;
  for (int v = 0; v < (a->nb + 7) / 8; v++)
    for (uint32_t c = c0; c < c1; c += 8)
      for (int l = 0; l < 8; l++) {
        float w[8] = {0};
        if (8 * v + l < a->nb) memcpy(w, a->x + (size_t)(8 * v + l) * C + c, sizeof w);
        for (int j = 0; j < 8; j++) PX[((size_t)v * C + c + j) * 8 + l] = w[j];
      }
}

// Prompt kernel on the TR rows r.. for nb tokens of PX: each TR x TC tile decoded once into fp32, then pk() over
// SC-column sub-blocks; the rows' escaped weights are added to the accumulators (8 tokens per FMA), then go to y.
static void prompt_rows(const Mat *m, uint32_t r, float *y, int nb) {
  static _Thread_local _Alignas(64) float ws[TR * TC];
  static _Thread_local _Alignas(64) __m256 st[TR][PB / 8];
  uint32_t R = m->rows, C = m->cols;
  int nv = (nb + 7) / 8;
  const uint8_t *p = m->w + r * row12(C);
  __m256 w[8];
  memset(st, 0, sizeof st);
  for (uint32_t c = 0; c < C; c += TC) {
    for (int i = 0; i < TR; i++)
      for (int j = 0; j < TC; j += BW, p += B12) {
        dec(p, m->lut, w);
        for (int k = 0; k < 8; k++) _mm256_store_ps(ws + i * TC + j + 8 * k, w[k]);
      }
    for (int s = 0; s < TC; s += SC)
      for (int v = 0; v < nv; v += VB)
        for (int i = 0; i < TR; i += RB) {
          const float *wi = ws + i * TC + s, *xv = PX + ((size_t)v * C + c + s) * 8;
          switch (nv - v) {
            case 1: pk(wi, xv, (size_t)C * 8, &st[i][v], 1); break;
            case 2: pk(wi, xv, (size_t)C * 8, &st[i][v], 2); break;
            default: pk(wi, xv, (size_t)C * 8, &st[i][v], VB);
          }
        }
  }
  for (uint32_t e = first_exc(m, r * C); e < m->ne && m->ei[e] < (r + TR) * C; e++) {
    const __m256 ev = _mm256_set1_ps(m->ev[e]);
    __m256 *o = st[m->ei[e] / C - r];
    const float *xc = PX + (size_t)(m->ei[e] % C) * 8;
    for (int v = 0; v < nv; v++) o[v] = _mm256_fmadd_ps(ev, _mm256_load_ps(xc + (size_t)v * C * 8), o[v]);
  }
  for (int i = 0; i < TR; i++)
    for (int b = 0; b < nb; b++) y[b * R + r + i] = ((const float *)st[i])[b];
}

static void mm_job(int t, void *arg) {
  const MM *a = arg;
  const Mat *m = a->m;
  const float *x = a->x;
  float *y = a->y, *glu = a->glu;
  int nb = a->nb;
  uint32_t R = m->rows, C = m->cols;
  const uint32_t tile = ~(uint32_t)(TR - 1);  // row ranges in whole TR-row blocks
  uint32_t r0 = (uint64_t)R * t / NT & tile, r1 = t == NT - 1 ? R : (uint64_t)R * (t + 1) / NT & tile;
  for (uint32_t r = r0; r < r1 && m->q5; r++)  // int5 (draft head): row by row
    for (int b = 0; b < nb; b += 4) {
      const uint8_t *p = m->w + r * row5(C);
#define Q5(G) coarse(p, C, x + b * C, y + b * R + r, R, G)
      G_SWITCH(Q5)
    }
  for (uint32_t r = r0; r < r1 && a->prompt; r += TR) prompt_rows(m, r, y, nb);
  for (uint32_t r = r0; r < r1 && !m->q5 && !a->prompt; r += TR)  // 12-bit: one TR-row block of tiles at a time
    for (int b = 0; b < nb; b += 4) {
      __m256 acc[TR][4][2] = {0};
      const uint8_t *p = m->w + r * row12(C);
      for (uint32_t c = 0; c < C; c += TC)
        for (int i = 0; i < TR; i++, p += row12(TC)) {
#define TILE(G) part(p, p + row12(TC), m->lut, x + b * C + c, C, acc[i], G)
          G_SWITCH(TILE)
        }
      for (int i = 0; i < TR; i++)
        for (int g = 0; g < nb - b && g < 4; g++)
          y[(b + g) * R + r + i] = hsum(_mm256_add_ps(acc[i][g][0], acc[i][g][1]));
    }
  for (uint32_t i = first_exc(m, r0 * C); !a->prompt && i < m->ne && m->ei[i] < r1 * C; i++)  // escaped weights
    for (int b = 0; b < nb; b++) y[b * R + m->ei[i] / C] += m->ev[i] * x[b * C + m->ei[i] % C];
  if (glu)
    for (int b = 0; b < nb; b++)
      for (uint32_t r = r0; r < r1; r += 2) glu[b * R / 2 + r / 2] = silu(y[b * R + r]) * y[b * R + r + 1];
}

static void mm(const Mat *m, const float *x, float *y, int nb, float *glu, int prompt) {
  MM a = {m, x, y, nb, glu, prompt && !m->q5};
  if (a.prompt) run(pack_job, &a);
  run(mm_job, &a);
}

// ---- Model state --------------------------------------------------------------------------------------------------

static Mat E, Wqkv[NL], Wo[NL], W13[NL], W2[NL];  // tied embedding/output matrix, layers (W13: gate/up interleaved)
static const float *rope_freq, *attn_norm[NL], *mlp_norm[NL], *out_norm;
static const uint32_t *voff;  // token i's bytes: vbytes[voff[i] .. voff[i + 1])
static const uint8_t *vbytes;
static const uint8_t *HQW;    // output-head screening copy: V int5 rows of HQ bytes
static const float *HC;       // per row: ||w - w_hat|| (1 + 1e-6) + 5e-4 ||w|| (the fp32 margin of the exact recompute)
static const float *HW;       // per row: ||w_hat|| (1 + 1e-6)
static int has_draft;
static const int CAPTURE[3] = {2, 8, 13};  // EAGLE-3 low/mid/high features: hidden states entering these layers
// activations, KV cache, draft features: cache-line aligned, so no 32-byte vector load crosses a line
static _Alignas(64) float X[PB * D], XN[PB * D], Q[PB * QKV], AO[PB * D], GU[PB * 2 * F], HB[PB * F], LG[MB * V];
// KV cache per KV head, each contiguous over positions (one stream per head); TK = T + MB: a draft chain runs ahead
enum { KVH = KV / HD, QPK = NH / KVH, TK = T + MB };  // KV heads, query heads per KV head, cache slots
_Static_assert(QPK % 2 == 0, "attn() accumulates the values of two query heads at a time");
static _Alignas(64) float KC[NL][KVH][TK][HD], VC[NL][KVH][TK][HD], H3[PB * 3 * D];

static void embed(int tok, float *o) {
  __m256 w[8];
  for (int c = 0; c < D; c += BW) {
    dec(E.w + tok * row12(D) + row12(c), E.lut, w);
    for (int k = 0; k < 8; k++) _mm256_storeu_ps(o + c + 8 * k, w[k]);
  }
  for (uint32_t i = first_exc(&E, (uint32_t)tok * D); i < E.ne && E.ei[i] < (uint32_t)(tok + 1) * D; i++)
    o[E.ei[i] % D] = E.ev[i];
}

static void rmsnorm(float *o, const float *x, const float *g) {
  __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();  // sum of squares: two 8-lane FMA chains
  for (int i = 0; i < D; i += 16) {
    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), _mm256_loadu_ps(x + i), a0);
    a1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 8), _mm256_loadu_ps(x + i + 8), a1);
  }
  const __m256 s = _mm256_set1_ps(1.0f / sqrtf(hsum(_mm256_add_ps(a0, a1)) / D + 1e-5f));  // eps = rms_norm_eps
  for (int i = 0; i < D; i += 8)
    _mm256_storeu_ps(o + i, _mm256_mul_ps(_mm256_loadu_ps(g + i), _mm256_mul_ps(_mm256_loadu_ps(x + i), s)));
}

static void rope_tab(int pos, float cs[HD]) {  // cos (first half) and sin (second half) of position pos
  for (int i = 0; i < HD / 2; i++) cs[i] = cosf(pos * rope_freq[i]), cs[i + HD / 2] = sinf(pos * rope_freq[i]);
}

static void rope(float *v, const float *cs) {  // rotate_half convention: pairs (i, i + HD/2), llama3-scaled freqs
  for (int i = 0; i < HD / 2; i++) {
    float c = cs[i], s = cs[i + HD / 2], a = v[i], b = v[i + HD / 2];
    v[i] = a * c - b * s;
    v[i + HD / 2] = b * c + a * s;
  }
}

// ---- Exact output head --------------------------------------------------------------------------------------------
// coarse_job computes the int5 x int8 screen values of its row range and, while they are cache-hot, per token its
// best lower bound lb and its rows whose upper bound reaches lb (a superset of the global candidates, as lb <= the
// global one). Only rows whose upper bound reaches the global best lower bound can be the argmax; rescore_job
// recomputes those exactly (row-major 12-bit weights + exceptions, rows()), each thread its own in ascending order,
// and the first maximum wins.

enum { NC = 64 };  // candidate rows kept per thread and token (more: argmax rescans that thread's range)
static struct { float lb; int n; uint32_t row[NC]; } CAND[MAXT][MB];
static float XNRM[MB], XB[MB];  // ||x_b||; the bound's factor for HW (see head_prep)
static _Alignas(64) int8_t XQ[MB * D];  // x_b quantized to int8 per BW-block, in the int5 slot order
static float XSC[MB * (D / BW)];        // their scales: x_hat = XSC * XQ
static const float UFLOW = 0x1p-100f;  // covers fp32 underflow, which relative error terms do not (README)
static const float GSCR = 2.3e-6f;      // >= gamma_36: the screen's fp32 rounding, rel. to ||w_hat|| ||x_hat||
static const double UP = 1 + 1e-6;      // rounds a double up past its conversion to fp32

static int slot_of(int c) {  // column within a BW-block -> its slot in the int5/12-bit layout (convert.py, widen())
  static const int8_t lane[8] = {0, 2, 4, 6, 16, 18, 20, 22}, off[4] = {0, 1, 8, 9};
  return lane[c % 8] + off[c / 8 % 4] + c / 32 * 32;
}

// The final features x of token b (in XN) for the screen: the int8 copy x_hat, XNRM >= ||x|| and XB = XE + GSCR
// (||x|| + XE) + UFLOW with XE >= ||x - x_hat||, both norms in double (exact squares, no underflow) and rounded up.
// Row bound: |exact logit - screen value| <= HC * ||x|| + HW * XB (README).
static void head_prep(int b) {
  const float *x = XN + b * D;
  double s = 0, err = 0;
  for (int i = 0; i < D; i++) s += (double)x[i] * x[i];
  XNRM[b] = (float)(sqrt(s) * UP);
  for (int g = 0; g < D / BW; g++) {
    float mx = 0;
    for (int c = 0; c < BW; c++) mx = fmaxf(mx, fabsf(x[g * BW + c]));
    float sc = mx / 127, inv = mx > UFLOW ? 127 / mx : 0;  // tiny blocks: r = 0 (still in XE)
    XSC[b * (D / BW) + g] = sc;
    for (int c = 0; c < BW; c++) {
      long r = lrintf(x[g * BW + c] * inv);
      r = r < -127 ? -127 : r > 127 ? 127 : r;
      XQ[b * D + g * BW + slot_of(c)] = (int8_t)r;
      double e = (double)x[g * BW + c] - (double)sc * r;  // exact
      err += e * e;
    }
  }
  float xe = (float)(sqrt(err) * UP);
  XB[b] = xe + GSCR * (XNRM[b] + xe) + UFLOW;
}

// The int5 head row p times the int8 features of G <= 4 tokens: per block an exact integer dot product of magnitudes
// (u8), signs (applied to x with vpsignb, +-1) and x (s8): vpmaddubsw pairs <= 3810 (no saturation), their int16
// sum <= 7620, vpmaddwd lanes <= 15240 < 2^24 (exact in fp32), then one fp32 FMA with the two scales.
static inline __attribute__((always_inline)) void iscreen(const uint8_t *p, const int8_t *xq, const float *xs,
                                                          float *out, uint32_t ostride, int G) {
  const __m256i nibble = _mm256_set1_epi8(15), one = _mm256_set1_epi8(1), ones16 = _mm256_set1_epi16(1),
                bit = _mm256_set1_epi64x(0x8040201008040201),  // byte i of each 8: 1 << i
                spread = _mm256_setr_epi64x(0, 0x0101010101010101, 0x0202020202020202, 0x0303030303030303);
  const uint32_t *sgn = (const uint32_t *)(p + D / 2);
  const uint16_t *sc = (const uint16_t *)(p + D / 2 + D / 8);
  __m256 tot[4] = {0};
  for (int g = 0; g < D / BW; g++) {
    const __m256i mag = _mm256_load_si256((const void *)(p + 32 * g));
    const __m256i k[2] = {_mm256_and_si256(mag, nibble), _mm256_and_si256(_mm256_srli_epi16(mag, 4), nibble)};
    __m256i sm[2];
    for (int i = 0; i < 2; i++) {
      __m256i bits = _mm256_shuffle_epi8(_mm256_set1_epi32(sgn[2 * g + i]), spread);  // sign bit j -> byte j
      sm[i] = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(bits, bit), bit), one);  // -1 or +1
    }
    const float swg = _cvtsh_ss(sc[g]);
    for (int b = 0; b < G; b++) {
      const int8_t *x = xq + b * D + g * BW;
      __m256i a = _mm256_maddubs_epi16(k[0], _mm256_sign_epi8(_mm256_load_si256((const void *)x), sm[0]));
      const __m256i x1 = _mm256_sign_epi8(_mm256_load_si256((const void *)(x + 32)), sm[1]);
      a = _mm256_add_epi16(a, _mm256_maddubs_epi16(k[1], x1));
      const __m256 f = _mm256_cvtepi32_ps(_mm256_madd_epi16(a, ones16));
      tot[b] = _mm256_fmadd_ps(f, _mm256_set1_ps(swg * xs[b * (D / BW) + g]), tot[b]);
    }
  }
  for (int b = 0; b < G; b++) out[b * ostride] = hsum(tot[b]);
}

// |exact logit - screen value| of row r for token b is at most this (README, exact output layer)
static inline float bound(uint32_t r, int b) { return HC[r] * XNRM[b] + HW[r] * XB[b]; }

static void coarse_job(int t, void *arg) {
  int nb = *(int *)arg;
  uint32_t r0 = (uint64_t)V * t / NT, r1 = (uint64_t)V * (t + 1) / NT;
  for (uint32_t r = r0; r < r1; r++)
    for (int b = 0; b < nb; b += 4) {
#define HEAD(G) iscreen(HQW + (size_t)r * HQ, XQ + b * D, XSC + b * (D / BW), LG + b * V + r, V, G)
      G_SWITCH(HEAD)
    }
  for (int b = 0; b < nb; b++) {
    const float *lg = LG + b * V;
    float lb = -INFINITY;
    for (uint32_t r = r0; r < r1; r++) {
      float low = lg[r] - bound(r, b);
      lb = low > lb ? low : lb;
    }
    CAND[t][b].lb = lb;
    CAND[t][b].n = 0;
    for (uint32_t r = r0; r < r1; r++)
      if (lg[r] + bound(r, b) >= lb && CAND[t][b].n++ < NC) CAND[t][b].row[CAND[t][b].n - 1] = r;
  }
}

// Each thread recomputes exactly those of its candidates whose upper bound reaches the global best lower bound, in
// ascending row order, and keeps its first maximum; argmax() takes the first maximum over the threads in order.
static struct { float v; int k; } BEST[MAXT][MB];

static void rescore_job(int t, void *arg) {
  int nb = *(int *)arg;
  for (int b = 0; b < nb; b++) {
    const float *x = XN + b * D, *lg = LG + b * V;
    float lb = -INFINITY, best = -INFINITY;
    int k = -1;
    for (int u = 0; u < NT; u++) lb = CAND[u][b].lb > lb ? CAND[u][b].lb : lb;
    uint32_t n = CAND[t][b].n, r = (uint64_t)V * t / NT, r1 = (uint64_t)V * (t + 1) / NT;
    for (uint32_t j = 0; n <= NC ? j < n : r < r1; j++, r++) {  // candidate list, or its whole range if it overflowed
      uint32_t i = n <= NC ? CAND[t][b].row[j] : r;
      if (lg[i] + bound(i, b) < lb) continue;
      const uint8_t *p = E.w + i * row12(D);
      float v;
      rows(p, p + row12(D), E.lut, x, D, &v, 1, 1);
      for (uint32_t e = first_exc(&E, i * D); e < E.ne && E.ei[e] < (i + 1) * D; e++) v += E.ev[e] * x[E.ei[e] % D];
      if (v > best) best = v, k = i;
    }
    BEST[t][b].v = best, BEST[t][b].k = k;
  }
}

static void head(int nb) {  // greedy tokens of the nb feature rows in XN (after head_prep): then argmax(b)
  run(coarse_job, &nb);
  run(rescore_job, &nb);
}

static int argmax(int b) {  // threads own ascending row ranges: the first strict maximum in thread order is the first
  int k = 0;
  float best = -INFINITY;
  for (int t = 0; t < NT; t++)
    if (BEST[t][b].k >= 0 && BEST[t][b].v > best) best = BEST[t][b].v, k = BEST[t][b].k;
  return k;
}

// ---- Attention and layers -----------------------------------------------------------------------------------------
// A pass runs a token tree of nb nodes (par[i] < i; par == 0 means a chain). Node i writes cache slot pos + i and sits
// at position pos + depth(i). It attends to the context slots < pos, then to its ancestors from the root down; for a
// chain those are slots pos..pos+i, the sequential order, so tree, batched and sequential results are bitwise equal.

static int depth(const int *par, int i) {
  if (!par) return i;
  int d = 0;
  for (; i > 0; i = par[i]) d++;
  return d;
}

typedef struct { int nb, pos; float (*kc)[TK][HD], (*vc)[TK][HD]; const int *par; } AT;
static void attn(int t, void *arg) {  // one (token, KV head) per task: its QPK query heads share each K and V row
  const AT *a = arg;
  static _Thread_local float att[QPK][TK];
  for (int bg = t; bg < a->nb * KVH; bg += NT) {
    int b = bg / KVH, g = bg % KVH, d = depth(a->par, b), n = a->pos + d + 1, sl[PB];
    for (int i = b, k = d; k >= 0; i = a->par ? a->par[i] : i - 1, k--) sl[k] = a->pos + i;  // ancestor slots
#define SLOT(j) ((j) < a->pos ? (j) : sl[(j) - a->pos])
    const float *q0 = Q + b * QKV + g * QPK * HD;
    float mx[QPK], sum[QPK];
    for (int h = 0; h < QPK; h++) mx[h] = -INFINITY, sum[h] = 0;
    for (int j = 0; j < n; j++) {  // scores: HD-wide dot products, two 8-lane FMA chains per head
      const float *k = a->kc[g][SLOT(j)];
      for (int h = 0; h < QPK; h++) {
        const float *q = q0 + h * HD;
        __m256 s0 = _mm256_mul_ps(_mm256_loadu_ps(q), _mm256_loadu_ps(k));
        __m256 s1 = _mm256_mul_ps(_mm256_loadu_ps(q + 8), _mm256_loadu_ps(k + 8));
        for (int i = 16; i < HD; i += 16) {
          s0 = _mm256_fmadd_ps(_mm256_loadu_ps(q + i), _mm256_loadu_ps(k + i), s0);
          s1 = _mm256_fmadd_ps(_mm256_loadu_ps(q + i + 8), _mm256_loadu_ps(k + i + 8), s1);
        }
        att[h][j] = hsum(_mm256_add_ps(s0, s1)) * 0.125f;  // 1/sqrt(HD)
        mx[h] = att[h][j] > mx[h] ? att[h][j] : mx[h];
      }
    }
    for (int h = 0; h < QPK; h++)
      for (int j = 0; j < n; j++) sum[h] += att[h][j] = expf(att[h][j] - mx[h]);
    for (int h = 0; h < QPK; h += 2) {  // softmax-weighted values, two heads' accumulators in registers
      __m256 o[2][HD / 8];
      for (int e = 0; e < 2; e++)
        for (int i = 0; i < HD / 8; i++) o[e][i] = _mm256_setzero_ps();
      for (int j = 0; j < n; j++) {
        const float *v = a->vc[g][SLOT(j)];
        for (int e = 0; e < 2; e++) {
          const __m256 p = _mm256_set1_ps(att[h + e][j] / sum[h + e]);
          for (int i = 0; i < HD / 8; i++) o[e][i] = _mm256_fmadd_ps(p, _mm256_loadu_ps(v + 8 * i), o[e][i]);
        }
      }
      for (int e = 0; e < 2; e++)
        for (int i = 0; i < HD / 8; i++) _mm256_storeu_ps(AO + b * D + (g * QPK + h + e) * HD + 8 * i, o[e][i]);
    }
  }
}

// One Llama layer on the residual stream X for nb tree nodes at positions pos..: attention, then the SwiGLU MLP.
// xin: the qkv input if it is not rmsnorm(X, g1) (the draft layer has its own 2D-wide input). A pass without a tree
// (par == 0) is a prompt chunk: its matrices use the prompt kernel.
static void layer(const Mat *wqkv, const Mat *wo, const Mat *w13, const Mat *w2, const float *g1, const float *g2,
                  float (*kc)[TK][HD], float (*vc)[TK][HD], int nb, int pos, const float *xin, const int *par) {
  if (!xin)
    for (int b = 0; b < nb; b++) rmsnorm(XN + b * D, X + b * D, g1);
  mm(wqkv, xin ? xin : XN, Q, nb, 0, !par);
  for (int b = 0; b < nb; b++) {
    float *q = Q + b * QKV, cs[HD];
    rope_tab(pos + depth(par, b), cs);
    for (int h = 0; h < NH; h++) rope(q + h * HD, cs);
    for (int h = 0; h < KVH; h++) rope(q + D + h * HD, cs);
    for (int g = 0; g < KVH; g++) {
      memcpy(kc[g][pos + b], q + D + g * HD, sizeof kc[0][0]);
      memcpy(vc[g][pos + b], q + D + KV + g * HD, sizeof vc[0][0]);
    }
  }
  AT at = {nb, pos, kc, vc, par};
  run(attn, &at);
  mm(wo, AO, XN, nb, 0, !par);
  for (int i = 0; i < nb * D; i++) X[i] += XN[i];
  for (int b = 0; b < nb; b++) rmsnorm(XN + b * D, X + b * D, g2);
  mm(w13, XN, GU, nb, HB, !par);
  mm(w2, HB, XN, nb, 0, !par);
  for (int i = 0; i < nb * D; i++) X[i] += XN[i];
}

// Runs the nb-node token tree par (0: a chain at positions pos..pos+nb-1). For the last nout nodes only (a prompt
// needs at most its final one) writes the greedy token after the node into out[] and leaves its features in XN.
static void forward(const int *tok, int nb, int pos, int *out, const int *par, int nout) {
  for (int b = 0; b < nb; b++) embed(tok[b], X + b * D);
  for (int l = 0; l < NL; l++) {
    for (int k = 0; k < 3 && has_draft; k++)  // draft input: hidden states entering the CAPTURE layers
      if (l == CAPTURE[k])
        for (int b = 0; b < nb; b++) memcpy(H3 + (3 * b + k) * D, X + b * D, D * sizeof *X);
    layer(&Wqkv[l], &Wo[l], &W13[l], &W2[l], attn_norm[l], mlp_norm[l], KC[l], VC[l], nb, pos, 0, par);
  }
  for (int b = 0; b < nout; b++) {
    rmsnorm(XN + b * D, X + (nb - nout + b) * D, out_norm);
    head_prep(b);
  }
  if (nout) head(nout);
  for (int b = 0; b < nout; b++) out[nb - nout + b] = argmax(b);
}

// ---- EAGLE-3 draft head (optional) --------------------------------------------------------------------------------
// Drafts are only proposals: forward() verifies every one, so they never change the output. Input feature =
// fc(target hidden states entering the CAPTURE layers); one layer whose qkv input is cat(rmsnorm(embed(next token)),
// rmsnorm(hidden)) and whose residual is the hidden state; lm_head over the NV most frequent tokens. All draft weights
// are int5 (drafts may be approximate). Chained steps feed the draft its own output hidden state.

enum { FD = 4096 };  // draft MLP width
static int NV;       // draft vocabulary size (from the model file)
static struct {
  Mat fc, qkv, o, w13, w2;
  const float *n_in, *n_hid, *n_post, *n_out;
  const uint8_t *head;
  const int32_t *vocab;
} DR;
static float DKC[KVH][TK][HD], DVC[KVH][TK][HD];  // the draft layer's KV cache

// Per thread: logits of its row range, then their top 2 (the first two of a stable descending sort, so merging the
// threads' pairs in row order gives the global pair) and the softmax partial sum relative to its own maximum.
static struct { int k, k2; float sum; } DTOP[MAXT];

static void top2(int r, int *k, int *k2) {  // fold row r into (k, k2): ties keep the earlier row first
  if (LG[r] > LG[*k]) *k2 = *k, *k = r;
  else if (*k2 < 0 || LG[r] > LG[*k2]) *k2 = r;
}

static void dhead_job(int t, void *x) {
  int r0 = (uint64_t)NV * t / NT, r1 = (uint64_t)NV * (t + 1) / NT, k = r0, k2 = -1;
  for (int r = r0; r < r1; r++) coarse(DR.head + (size_t)r * HQ, D, x, LG + r, 1, 1);
  for (int r = r0 + 1; r < r1; r++) top2(r, &k, &k2);
  float sum = 0;
  for (int r = r0; r < r1; r++) sum += expf(LG[r] - LG[k]);
  DTOP[t].k = k, DTOP[t].k2 = k2, DTOP[t].sum = sum;
}

// One draft step for nb (hidden state in X, next token) pairs at positions pos..; leaves the output hidden states in
// X. Unless p == 0 (cache filling only), returns the draft token after the last pair with its draft probability *p,
// and the runner-up *t2 with *p2.
static int dstep(const int *tok, int nb, int pos, float *p, int *t2, float *p2) {
  static _Alignas(64) float in[PB * 2 * D];
  for (int b = 0; b < nb; b++) {
    embed(tok[b], XN + b * D);  // XN as scratch
    rmsnorm(in + 2 * D * b, XN + b * D, DR.n_in);
    rmsnorm(in + 2 * D * b + D, X + D * b, DR.n_hid);
  }
  layer(&DR.qkv, &DR.o, &DR.w13, &DR.w2, 0, DR.n_post, DKC, DVC, nb, pos, in, 0);
  if (!p) return -1;
  rmsnorm(XN, X + (nb - 1) * D, DR.n_out);
  run(dhead_job, XN);
  int k = DTOP[0].k, k2 = DTOP[0].k2;
  for (int t = 1; t < NT; t++) {  // each thread's pair, in row order
    int first = DTOP[t].k, second = DTOP[t].k2;
    if (second >= 0 && second < first) first = second, second = DTOP[t].k;
    top2(first, &k, &k2);
    if (second >= 0) top2(second, &k, &k2);
  }
  float sum = 0;
  for (int t = 0; t < NT; t++) sum += DTOP[t].sum * expf(LG[DTOP[t].k] - LG[k]);
  *p = 1 / sum;
  *p2 = expf(LG[k2] - LG[k]) / sum;
  *t2 = DR.vocab[k2];
  return DR.vocab[k];
}

// Draft tree after the nb positions pos.. of the last pass (their features are in H3, their next tokens in next),
// EAGLE-2 style: a chain of top-1 drafts (up to K steps, while its joint probability stays >= TAU) gives 2 candidates
// per depth for free, the chain token and the runner-up; the K most probable candidates with probability >= TAU form
// the tree (a parent is never less probable than its child, so the selection is always a valid tree). Fills node
// tokens d[i] and parents par[i] (node numbers: 0 = the root, i + 1 = d[i]); returns the node count.
enum { K_DEFAULT = 3, NEW_DEFAULT = 256 };  // default draft size and new-token limit (command line)
static const float TAU = 0.1f;               // tuned: K = 3, TAU = 0.1 best over K = 1..3 x TAU = 0..0.3
static int drafts(const int *next, int nb, int pos, int K, int *d, int *par) {
  float pr[2 * MB], chain = 1, p, p2;  // candidate 2k: chain token at depth k; 2k + 1: runner-up at depth k
  int tk[2 * MB], sel[2 * MB] = {0}, node[MB], L = 0, n = 0;
  if (!has_draft) return 0;
  mm(&DR.fc, H3, X, nb, 0, 0);  // fused features straight into the residual stream
  if (!K) {                  // prompt chunks: only fill the draft's cache
    dstep(next, nb, pos, 0, 0, 0);
    return 0;
  }
  while (L < K && chain >= TAU) {
    if (L == 1 && nb > 1) memcpy(X, X + (nb - 1) * D, D * sizeof *X);  // the chain continues from the last row only
    tk[2 * L] = dstep(L ? tk + 2 * L - 2 : next, L ? 1 : nb, L ? pos + nb + L - 1 : pos, &p, tk + 2 * L + 1, &p2);
    pr[2 * L + 1] = chain * p2;
    pr[2 * L] = chain *= p;
    L++;
  }
  for (int m = 0; m < K; m++) {  // the K most probable candidates (ties: lower index, i.e. parents first)
    int best = -1;
    for (int i = 0; i < 2 * L; i++)
      if (!sel[i] && pr[i] >= TAU && (best < 0 || pr[i] > pr[best])) best = i;
    if (best < 0) break;
    sel[best] = 1;
  }
  for (int k = 0; k < L; k++)  // number the tree by depth, chain node first
    for (int i = 2 * k; i < 2 * k + 2; i++)
      if (sel[i]) {
        d[n] = tk[i];
        par[n] = k ? node[k - 1] : 0;
        n++;
        if (i == 2 * k) node[k] = n;  // node number of the chain token at depth k
      }
  return n;
}

// ---- Tokenizer: Llama 3 pre-tokenizer regex (non-ASCII bytes count as letters) + rank-ordered byte BPE ------

enum { HT = 1 << 18 };
static uint32_t ht[HT];  // open addressing, token bytes -> token id + 1

static uint32_t hsh(const uint8_t *s, int n) {  // FNV-1a
  uint32_t h = 2166136261u;
  while (n--) h = (h ^ *s++) * 16777619u;
  return h;
}

static int look(const uint8_t *s, int n) {  // token id of the bytes s[0..n), or -1
  for (uint32_t i = hsh(s, n);; i++) {
    uint32_t t = ht[i & (HT - 1)];
    if (!t) return -1;
    if (voff[t] - voff[t - 1] == (uint32_t)n && !memcmp(vbytes + voff[t - 1], s, n)) return t - 1;
  }
}

static int isL(int c) { return (unsigned)((c | 32) - 'a') < 26 || c >= 128; }  // letter
static int isN(int c) { return (unsigned)(c - '0') < 10; }                      // digit
static int isS(int c) { return c == ' ' || (unsigned)(c - 9) < 5; }             // whitespace
static int isP(int c) { return !isS(c) && !isL(c) && !isN(c); }                 // other
static int isNL(int c) { return c == '\r' || c == '\n'; }

// length of the next pre-token: the alternatives of the Llama 3 regex, in order
static int chunk(const uint8_t *s, int n) {
  if (s[0] == '\'' && n > 1) {  // 's 't 'm 'd 're 've 'll
    int c = s[1] | 32, d = n > 2 ? s[2] | 32 : 0;
    if (c == 's' || c == 't' || c == 'm' || c == 'd') return 2;
    if ((c == 'r' && d == 'e') || (c == 'v' && d == 'e') || (c == 'l' && d == 'l')) return 3;
  }
  int i = !isL(s[0]) && !isN(s[0]) && !isNL(s[0]) && n > 1 && isL(s[1]);  // optional non-letter prefix + letters
  if (isL(s[i])) {
    while (i < n && isL(s[i])) i++;
    return i;
  }
  if (isN(s[0])) {  // 1-3 digits
    while (i < n && i < 3 && isN(s[i])) i++;
    return i;
  }
  i = s[0] == ' ' && n > 1 && isP(s[1]);  // optional space + other characters + newlines
  if (i || isP(s[0])) {
    while (i < n && isP(s[i])) i++;
    while (i < n && isNL(s[i])) i++;
    return i;
  }
  int j = 0, nl = 0;  // whitespace: up to its last newline; else all but the last space before text
  while (j < n && isS(s[j])) {
    j++;
    if (isNL(s[j - 1])) nl = j;
  }
  return nl ? nl : j == n || j == 1 ? j : j - 1;
}

static int encode(const char *txt, int *out, int cap) {  // -> token count, or -1 if more than cap
  const uint8_t *s = (const uint8_t *)txt;
  int n = strlen(txt), k = 0;
  while (n > 0) {
    int c = chunk(s, n), m = 0;
    int *st = malloc(2 * c * sizeof *st), *ln = st + c;  // pieces: start, length (pre-tokens can be long)
    if (!st) {
      fprintf(stderr, "out of memory\n");
      exit(1);
    }
    for (int i = 0; i < c; i++) st[m] = i, ln[m++] = 1;
    for (;;) {  // merge the adjacent pair whose result has the lowest rank (= token id)
      int best = -1, bi = 0;
      for (int i = 0; i + 1 < m; i++) {
        int t = look(s + st[i], ln[i] + ln[i + 1]);
        if (t >= 0 && (best < 0 || t < best)) best = t, bi = i;
      }
      if (best < 0) break;
      ln[bi] += ln[bi + 1];
      memmove(st + bi + 1, st + bi + 2, (m - bi - 2) * sizeof *st);
      memmove(ln + bi + 1, ln + bi + 2, (m - bi - 2) * sizeof *ln);
      m--;
    }
    for (int i = 0; i < m && k <= cap; i++) out[k++] = look(s + st[i], ln[i]);
    free(st);
    s += c, n -= c;
    if (k > cap) return -1;
  }
  return k;
}

// ---- Loading ----------------------------------------------------------------------------------------------------
// The whole file goes into 2 MB pages (no TLB/prefetch stalls every 4 KB), read with O_DIRECT: straight from disk
// into place, without a second copy in the page cache.

enum { HUGE_PAGE = 2 << 20, DIRECT_BLOCK = 4096, READ_CHUNK = 1 << 30 };

static int load(const char *path) {
  struct stat st;
  int fd = open(path, O_RDONLY | O_DIRECT);
  if (fd < 0) fd = open(path, O_RDONLY);  // filesystems without O_DIRECT
  if (fd < 0 || fstat(fd, &st)) return perror(path), 1;
  uint8_t *buf = mmap(0, st.st_size + 2 * HUGE_PAGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (buf == MAP_FAILED) return perror("mmap"), 1;
  buf = (uint8_t *)(((uintptr_t)buf + HUGE_PAGE - 1) & ~(uintptr_t)(HUGE_PAGE - 1));
  madvise(buf, st.st_size + HUGE_PAGE, MADV_HUGEPAGE);
  for (ssize_t o = 0, r; o < st.st_size; o += r) {  // O_DIRECT reads must be whole blocks
    size_t rest = st.st_size - o;
    size_t want = rest > READ_CHUNK ? READ_CHUNK : (rest + DIRECT_BLOCK - 1) & ~(size_t)(DIRECT_BLOCK - 1);
    if ((r = read(fd, buf + o, want)) <= 0) return perror("read"), 1;
  }
  close(fd);
  P = buf, PEND = buf + st.st_size;
  if (memcmp(take(4), MAGIC, 4))
    return fprintf(stderr, "%s: not a %s model file\n", path, MAGIC), 1;
  rope_freq = take(HD / 2 * sizeof *rope_freq);
  for (int l = 0; l < NL; l++) attn_norm[l] = take(D * 4), mlp_norm[l] = take(D * 4);
  out_norm = take(D * 4);
  voff = take((V + 1) * 4);
  vbytes = take(voff[V]);
  E = mat(V, D);
  HQW = take((size_t)V * HQ);
  HC = take(V * 4);
  HW = take(V * 4);
  for (int l = 0; l < NL; l++) Wqkv[l] = mat(QKV, D), Wo[l] = mat(D, D), W13[l] = mat(2 * F, D), W2[l] = mat(D, F);
  if (P < PEND) {  // optional draft head section
    // NV >= MAXT: every thread owns draft-head rows (dhead_job)
    if ((NV = *(const uint32_t *)take(4)) < MAXT || NV > V)
      return fprintf(stderr, "%s: bad draft vocabulary\n", path), 1;
    DR.vocab = take(NV * 4);
    DR.n_in = take(D * 4), DR.n_hid = take(D * 4), DR.n_post = take(D * 4), DR.n_out = take(D * 4);
    DR.fc = mat5(D, 3 * D), DR.qkv = mat5(QKV, 2 * D), DR.o = mat5(D, D), DR.w13 = mat5(2 * FD, D), DR.w2 = mat5(D, FD);
    DR.head = take((size_t)NV * HQ);
    has_draft = 1;
  }
  for (int i = 0; i < SPECIAL; i++)  // BPE lookup table over the ordinary tokens
    for (uint32_t h = hsh(vbytes + voff[i], voff[i + 1] - voff[i]);; h++)
      if (!ht[h & (HT - 1)]) {
        ht[h & (HT - 1)] = i + 1;
        break;
      }
  scan_cores();
  // all physical cores: best for decoding unless the CPU is power-limited (a laptop running hot: set TLM_THREADS)
  NT = getenv("TLM_THREADS") ? atoi(getenv("TLM_THREADS")) : NCORE;
  NT = NT < 1 ? 1 : NT > MAXT ? MAXT : NT;
  for (intptr_t t = 1; t < NT; t++) {
    pthread_t th;
    pthread_create(&th, 0, worker, (void *)t);
  }
  pin(0);
  return 0;
}

// ---- Generation ----------------------------------------------------------------------------------------------------

// Llama 3.2 chat template, single user turn. The system header is the official template's default: its fixed
// knowledge cutoff plus today's date (strftime_now("%d %b %Y")). Returns the token count, or -1 above PCAP.
static int chat_prompt(const char *user, int *ctx) {
  char sys[96], date[32];
  time_t now = time(0);
  strftime(date, sizeof date, "%d %b %Y", localtime(&now));
  snprintf(sys, sizeof sys, "\n\nCutting Knowledge Date: December 2023\nToday Date: %s\n\n", date);
  int n = 0, u;
  ctx[n++] = BOS;
  ctx[n++] = HDR_START, n += encode("system", ctx + n, 8), ctx[n++] = HDR_END;
  n += encode(sys, ctx + n, 64), ctx[n++] = EOT;
  ctx[n++] = HDR_START, n += encode("user", ctx + n, 8), ctx[n++] = HDR_END;
  n += encode("\n\n", ctx + n, 8);
  if ((u = encode(user, ctx + n, PCAP - n - 16)) < 0) return -1;  // 16: room for the template tokens that follow
  n += u, ctx[n++] = EOT;
  ctx[n++] = HDR_START, n += encode("assistant", ctx + n, 8), ctx[n++] = HDR_END;
  n += encode("\n\n", ctx + n, 8);
  return n;
}

// Greedy generation from the prompt ctx[0..n). Every pass verifies up to K drafted tokens (draft head if the model
// has one, else prompt lookup) and emits the accepted ones plus the model's own next token. Returns the new length m;
// *passes counts forward passes after the prompt. The prompt is processed from position `from` on: a caller that keeps
// ctx and extends it (multi-turn) passes the m - 1 of the previous call, as the cache then holds ctx[0..m - 1). The
// prompt kernel does not depend on how the prompt is split, but the cache's reply rows came from decode passes, which
// round differently: the result is token-identical to that of from = 0 (tui's check), not bitwise.
static int generate(int *ctx, int n, int from, int maxn, int K, void (*emit)(int), int *passes) {
  int np = n, a[PB], tok = 0, dr[MB], dpar[MB], ndr = 0, next[PB];
  K = K < 0 ? 0 : K > MB - 1 ? MB - 1 : K;  // a pass verifies at most MB tokens
  from = from < 0 ? 0 : from > np - 1 ? np - 1 : from;  // at least the last prompt token yields the first logits
  for (int i = from; i < np; i += PB) {                 // the prompt, in chunks
    int nb = np - i < PB ? np - i : PB;
    forward(ctx + i, nb, i, a, 0, i + nb == np);
    tok = a[nb - 1];
    for (int b = 0; b < nb; b++) next[b] = i + b + 1 < np ? ctx[i + b + 1] : tok;
    if (K) ndr = drafts(next, nb, i, i + nb == np ? K : 0, dr, dpar);  // earlier chunks only fill the draft cache
  }
  int pend[MB] = {tok}, npend = 1;  // tokens decided by the last pass, emitted one by one with the stop checks
  for (*passes = 0;;) {
    int stop = 0;
    for (int i = 0; i < npend && !stop; i++) {
      ctx[n++] = tok = pend[i];
      stop = tok == EOT || tok == EOM || tok == END_OF_TEXT || n - np >= maxn || n >= T - MB;
      if (tok < SPECIAL) emit(tok);
    }
    if (stop) return n;
    int f[MB] = {tok}, par[MB] = {-1}, nd = 0;  // the tree: root = last token, then the drafts
    for (; nd < ndr; nd++) f[1 + nd] = dr[nd], par[1 + nd] = dpar[nd];
    // prompt lookup (no draft head): the latest earlier occurrence of the last g = 3..1 tokens; draft what followed
    // it, at most g tokens (longer matches are stronger evidence) and at most K
    for (int g = 3; g > 0 && !nd && K && !has_draft; g--)
      for (int j = n - g - 1; j >= 0 && !nd; j--)
        if (!memcmp(ctx + j, ctx + n - g, g * sizeof *ctx))
          while (nd < K && nd < g && j + g + nd < n) f[1 + nd] = ctx[j + g + nd], par[1 + nd] = nd, nd++;
    forward(f, 1 + nd, n - 1, a, par, 1 + nd);
    ++*passes;
    int u = 0, path[MB] = {0}, m = 0;  // follow the model's own choices: accepted drafts, then its next token
    for (npend = 0;; pend[npend++] = f[u]) {
      int c = 1;
      while (c <= nd && !(par[c] == u && f[c] == a[u])) c++;
      if (c > nd) break;
      path[++m] = u = c;
    }
    pend[npend++] = a[u];
    for (int k = 1; k <= m; k++)  // accepted node -> the slot of its depth (KV rows, draft features)
      if (path[k] != k) {
        for (int l = 0; l < NL; l++) {
          for (int g = 0; g < KVH; g++) {
            memcpy(KC[l][g][n - 1 + k], KC[l][g][n - 1 + path[k]], sizeof KC[0][0][0]);
            memcpy(VC[l][g][n - 1 + k], VC[l][g][n - 1 + path[k]], sizeof VC[0][0][0]);
          }
        }
        memcpy(H3 + 3 * D * k, H3 + 3 * D * path[k], 3 * D * sizeof *H3);
      }
    for (int b = 0; b < npend; b++) next[b] = pend[b];  // re-feed the draft with true features of accepted positions
    ndr = K ? drafts(next, npend, n - 1, K, dr, dpar) : 0;
  }
}

static void print_token(int tok) {
  fwrite(vbytes + voff[tok], 1, voff[tok + 1] - voff[tok], stdout);
  fflush(stdout);
}

int main(int argc, char **argv) {
  if (argc < 3) return fprintf(stderr, "usage: %s model.tlm prompt [max_new_tokens=256] [max_draft=3]\n", argv[0]), 1;
  static int ctx[T];
  int passes, n;
  if (load(argv[1])) return 1;
  if ((n = chat_prompt(argv[2], ctx)) < 0) return fprintf(stderr, "prompt too long (at most %d tokens)\n", PCAP), 1;
  int maxn = argc > 3 ? atoi(argv[3]) : NEW_DEFAULT, K = argc > 4 ? atoi(argv[4]) : K_DEFAULT;
  generate(ctx, n, 0, maxn, K, print_token, &passes);
  putchar('\n');
  return 0;
}
