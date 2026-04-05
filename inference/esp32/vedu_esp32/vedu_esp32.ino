/*
 * VEDU-BITNET ESP32 INFERENCE ENGINE
 * ====================================
 * Bare-metal, multiply-free CNN inference for ESP32.
 *
 * This is a self-contained Arduino sketch that:
 *   1. Loads the ternary model from flash (PROGMEM)
 *   2. Runs the test image through the multiply-free pipeline
 *   3. Reports prediction + timing over Serial
 *
 * Fully dynamic: works with any VBN model (MNIST, CIFAR-10, etc.)
 * Buffer sizes are computed from layer metadata at runtime.
 *
 * Board: ESP32 (any variant) or ESP32-S3
 * Upload: Arduino IDE → Tools → Board → ESP32 Dev Module
 */

#include <cstdint>
#include <cstring>
#include <cmath>
#include "vedu_model_data.h"   // Auto-generated model in flash

// ============================================================
// Model data structures (parsed from flash)
// ============================================================

struct LayerData {
    uint8_t type;          // 0=conv, 1=linear
    uint32_t out_ch, in_ch, ksize, stride, padding;
    uint32_t out_features, in_features;
    // Pointers into parsed data (heap-allocated at setup)
    int8_t*  weights;
    float*   bias;
    float*   norm_gamma;
    float*   norm_beta;
    uint32_t num_weights;
    uint32_t num_bias;
    uint32_t num_gamma;
    uint32_t num_beta;
};

#define MAX_LAYERS 4

static LayerData g_layers[MAX_LAYERS];
static uint32_t  g_num_layers = 0;

// Test vector
static float*   g_test_input = nullptr;
static int32_t  g_test_expected = -1;
static bool     g_has_test = false;

// Input dimensions (derived from model + test vector)
static uint32_t g_input_floats = 0;

// ============================================================
// Flash reader helpers
// ============================================================

class FlashReader {
    const uint8_t* _data;
    uint32_t _pos;
    uint32_t _size;
public:
    FlashReader(const uint8_t* data, uint32_t size)
        : _data(data), _pos(0), _size(size) {}

    bool read_u8(uint8_t& v) {
        if (_pos + 1 > _size) return false;
#ifdef ESP32
        v = pgm_read_byte(&_data[_pos]);
#else
        v = _data[_pos];
#endif
        _pos += 1;
        return true;
    }

    bool read_u32(uint32_t& v) {
        if (_pos + 4 > _size) return false;
        // Read little-endian from flash
        uint8_t buf[4];
        for (int i = 0; i < 4; i++) {
#ifdef ESP32
            buf[i] = pgm_read_byte(&_data[_pos + i]);
#else
            buf[i] = _data[_pos + i];
#endif
        }
        memcpy(&v, buf, 4);
        _pos += 4;
        return true;
    }

    bool read_i32(int32_t& v) {
        return read_u32(*(uint32_t*)&v);
    }

    bool read_f32_array(float* out, uint32_t count) {
        if (_pos + count * 4 > _size) return false;
        for (uint32_t i = 0; i < count; i++) {
            uint8_t buf[4];
            for (int j = 0; j < 4; j++) {
#ifdef ESP32
                buf[j] = pgm_read_byte(&_data[_pos + j]);
#else
                buf[j] = _data[_pos + j];
#endif
            }
            memcpy(&out[i], buf, 4);
            _pos += 4;
        }
        return true;
    }

    bool read_packed_ternary(int8_t* out, uint32_t num_weights, uint32_t num_blocks) {
        uint32_t wi = 0;
        for (uint32_t b = 0; b < num_blocks; b++) {
            uint32_t block;
            if (!read_u32(block)) return false;
            for (int j = 0; j < 16 && wi < num_weights; j++, wi++) {
                uint8_t code = (block >> (j * 2)) & 0x03;
                if (code == 0x01)      out[wi] = +1;
                else if (code == 0x02) out[wi] = -1;
                else                    out[wi] = 0;
            }
        }
        return true;
    }

    bool check_magic(const char* expected, int len) {
        if (_pos + len > _size) return false;
        for (int i = 0; i < len; i++) {
#ifdef ESP32
            if (pgm_read_byte(&_data[_pos + i]) != (uint8_t)expected[i])
                return false;
#else
            if (_data[_pos + i] != (uint8_t)expected[i])
                return false;
#endif
        }
        _pos += len;
        return true;
    }

    uint32_t pos() const { return _pos; }
    uint32_t remaining() const { return _size - _pos; }
};

// ============================================================
// Parse model from flash
// ============================================================

static bool parse_model() {
    FlashReader r(VEDU_MODEL_DATA, VEDU_MODEL_SIZE);

    if (!r.check_magic("VBN1", 4)) {
        Serial.println("ERROR: Invalid magic");
        return false;
    }

    r.read_u32(g_num_layers);
    Serial.printf("Loading %u layers from flash...\n", g_num_layers);

    for (uint32_t i = 0; i < g_num_layers && i < MAX_LAYERS; i++) {
        LayerData& L = g_layers[i];
        r.read_u8(L.type);

        if (L.type == 0) { // Conv
            r.read_u32(L.out_ch);
            r.read_u32(L.in_ch);
            r.read_u32(L.ksize);
            r.read_u32(L.stride);
            r.read_u32(L.padding);
            L.out_features = L.out_ch;
            L.in_features = L.in_ch;
            Serial.printf("  L%u: Conv(%u->%u, %ux%u)\n",
                          i, L.in_ch, L.out_ch, L.ksize, L.ksize);
        } else { // Linear
            r.read_u32(L.out_features);
            r.read_u32(L.in_features);
            L.out_ch = L.out_features;
            L.in_ch = L.in_features;
            L.ksize = 1; L.stride = 1; L.padding = 0;
            Serial.printf("  L%u: Linear(%u->%u)\n",
                          i, L.in_features, L.out_features);
        }

        // Packed ternary weights
        uint32_t num_weights, num_blocks;
        r.read_u32(num_weights);
        r.read_u32(num_blocks);

        L.num_weights = num_weights;
        L.weights = (int8_t*)malloc(num_weights);
        if (!L.weights) {
            Serial.printf("  ERROR: malloc(%u) for weights failed!\n", num_weights);
            return false;
        }
        r.read_packed_ternary(L.weights, num_weights, num_blocks);

        // Bias
        r.read_u32(L.num_bias);
        L.bias = (float*)malloc(L.num_bias * 4);
        r.read_f32_array(L.bias, L.num_bias);

        // Norm gamma
        r.read_u32(L.num_gamma);
        L.norm_gamma = (float*)malloc(L.num_gamma * 4);
        r.read_f32_array(L.norm_gamma, L.num_gamma);

        // Norm beta
        r.read_u32(L.num_beta);
        L.norm_beta = (float*)malloc(L.num_beta * 4);
        r.read_f32_array(L.norm_beta, L.num_beta);

        uint32_t ram_bytes = num_weights + L.num_bias * 4
                           + L.num_gamma * 4 + L.num_beta * 4;
        Serial.printf("    RAM: %u bytes\n", ram_bytes);
    }

    // Test vector
    if (r.remaining() >= 4 && r.check_magic("TEST", 4)) {
        g_has_test = true;
        uint32_t inp_size;
        r.read_u32(inp_size);
        g_input_floats = inp_size;  // store for buffer allocation
        g_test_input = (float*)malloc(inp_size * 4);
        r.read_f32_array(g_test_input, inp_size);
        r.read_i32(g_test_expected);
        // Skip logits (we don't need them on ESP32)
        Serial.printf("Test vector loaded (%u floats, expected: %d)\n",
                      inp_size, g_test_expected);
    }

    return true;
}

// ============================================================
// Neural network operations
// ============================================================

// GroupNorm(1, C) on [C, H, W]
static void group_norm(float* data, int C, int H, int W,
                       const float* gamma, const float* beta) {
    const int total = C * H * W;
    const float eps = 1e-5f;

    double sum = 0.0;
    for (int i = 0; i < total; i++) sum += data[i];
    float mean = (float)(sum / total);

    double var_sum = 0.0;
    for (int i = 0; i < total; i++) {
        float d = data[i] - mean;
        var_sum += (double)d * d;
    }
    float inv_std = 1.0f / sqrtf((float)(var_sum / total) + eps);

    for (int c = 0; c < C; c++) {
        float g = gamma[c];
        float b = beta[c];
        for (int hw = 0; hw < H * W; hw++) {
            int idx = c * H * W + hw;
            data[idx] = (data[idx] - mean) * inv_std * g + b;
        }
    }
}

// LayerNorm(features) on [features]
static void layer_norm(float* data, int features,
                       const float* gamma, const float* beta) {
    const float eps = 1e-5f;

    double sum = 0.0;
    for (int i = 0; i < features; i++) sum += data[i];
    float mean = (float)(sum / features);

    double var_sum = 0.0;
    for (int i = 0; i < features; i++) {
        float d = data[i] - mean;
        var_sum += (double)d * d;
    }
    float inv_std = 1.0f / sqrtf((float)(var_sum / features) + eps);

    for (int i = 0; i < features; i++) {
        data[i] = (data[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

static void relu(float* data, int n) {
    for (int i = 0; i < n; i++) {
        if (data[i] < 0.0f) data[i] = 0.0f;
    }
}

// MaxPool2d(2): [C, H, W] → [C, H/2, W/2]
static void max_pool2d(const float* in, float* out,
                       int C, int H, int W) {
    int oH = H / 2, oW = W / 2;
    for (int c = 0; c < C; c++) {
        for (int oh = 0; oh < oH; oh++) {
            for (int ow = 0; ow < oW; ow++) {
                float mx = -1e30f;
                for (int kh = 0; kh < 2; kh++) {
                    for (int kw = 0; kw < 2; kw++) {
                        int ih = oh * 2 + kh;
                        int iw = ow * 2 + kw;
                        float v = in[c * H * W + ih * W + iw];
                        if (v > mx) mx = v;
                    }
                }
                out[c * oH * oW + oh * oW + ow] = mx;
            }
        }
    }
}

// ============================================================
// MULTIPLY-FREE CONVOLUTION (the core innovation)
// ============================================================

static void ternary_conv2d(
    const float* input, float* output,
    const int8_t* weights, const float* bias,
    int in_ch, int out_ch, int K,
    int H, int W, int stride, int pad)
{
    int oH = (H + 2 * pad - K) / stride + 1;
    int oW = (W + 2 * pad - K) / stride + 1;

    for (int oc = 0; oc < out_ch; oc++) {
        for (int oh = 0; oh < oH; oh++) {
            for (int ow = 0; ow < oW; ow++) {
                float acc = bias[oc];

                for (int ic = 0; ic < in_ch; ic++) {
                    for (int kh = 0; kh < K; kh++) {
                        for (int kw = 0; kw < K; kw++) {
                            int ih = oh * stride - pad + kh;
                            int iw = ow * stride - pad + kw;

                            if (ih < 0 || ih >= H || iw < 0 || iw >= W)
                                continue;

                            float pixel = input[ic * H * W + ih * W + iw];
                            int8_t w = weights[((oc * in_ch + ic) * K + kh) * K + kw];

                            // === MULTIPLY-FREE ===
                            if (w == 1) {
                                acc += pixel;
                            } else if (w == -1) {
                                acc -= pixel;
                            }
                            // w == 0: skip
                        }
                    }
                }
                output[oc * oH * oW + oh * oW + ow] = acc;
            }
        }
    }
}

// ============================================================
// MULTIPLY-FREE LINEAR LAYER
// ============================================================

static void ternary_linear(
    const float* input, float* output,
    const int8_t* weights, const float* bias,
    int in_features, int out_features)
{
    for (int o = 0; o < out_features; o++) {
        float acc = bias[o];
        for (int i = 0; i < in_features; i++) {
            int8_t w = weights[o * in_features + i];
            if (w == 1) {
                acc += input[i];
            } else if (w == -1) {
                acc -= input[i];
            }
        }
        output[o] = acc;
    }
}

// ============================================================
// Full forward pass — fully dynamic
// ============================================================
// Working buffers allocated once from PSRAM, sizes computed from model

static float* g_work_a = nullptr;  // current activation buffer
static float* g_work_b = nullptr;  // scratch buffer for conv/linear output
static uint32_t g_work_a_size = 0; // size in floats
static uint32_t g_work_b_size = 0;

// Derived from model + test vector
static int g_input_ch = 0;
static int g_input_h  = 0;
static int g_input_w  = 0;
static int g_num_logits = 0;

static bool alloc_buffers() {
    // Derive input spatial dims from test vector and first conv layer
    g_input_ch = g_layers[0].in_ch;
    // Count test input floats
    // g_test_input was allocated with inp_size floats
    // We stored that size; derive from test vector
    // Actually we need to know the test input size — store it globally
    // For now, compute from the model: in_ch is known,
    // total = in_ch * H * W. We need H. Let's store inp_size during parse.
    // (See parse_model — g_input_floats is set there)

    if (g_input_floats == 0 || g_input_ch == 0) {
        Serial.println("ERROR: Cannot derive input dims");
        return false;
    }

    int hw = g_input_floats / g_input_ch;
    g_input_h = (int)sqrtf((float)hw);
    g_input_w = g_input_h;

    Serial.printf("Input: %dx%dx%d (%u floats)\n",
                  g_input_ch, g_input_h, g_input_w, g_input_floats);

    // Compute max buffer size needed across all layers
    uint32_t max_buf = g_input_floats; // start with input size
    int cur_C = g_input_ch, cur_H = g_input_h, cur_W = g_input_w;

    for (uint32_t i = 0; i < g_num_layers; i++) {
        LayerData& L = g_layers[i];
        if (L.type == 0) { // Conv
            int oH = (cur_H + 2 * (int)L.padding - (int)L.ksize) / (int)L.stride + 1;
            int oW = (cur_W + 2 * (int)L.padding - (int)L.ksize) / (int)L.stride + 1;
            uint32_t conv_size = L.out_ch * oH * oW;
            if (conv_size > max_buf) max_buf = conv_size;
            // After pool
            int pH = oH / 2, pW = oW / 2;
            uint32_t pool_size = L.out_ch * pH * pW;
            if (pool_size > max_buf) max_buf = pool_size;
            cur_C = L.out_ch; cur_H = pH; cur_W = pW;
        } else { // Linear
            if (L.out_features > max_buf) max_buf = L.out_features;
            if (L.in_features > max_buf) max_buf = L.in_features;
        }
    }

    // Get num_logits from last layer
    g_num_logits = g_layers[g_num_layers - 1].out_features;

    // Allocate two buffers of max_buf size (ping-pong)
    g_work_a_size = max_buf;
    g_work_b_size = max_buf;

#ifdef ESP32
    g_work_a = (float*)ps_malloc(max_buf * sizeof(float));
    g_work_b = (float*)ps_malloc(max_buf * sizeof(float));
#else
    g_work_a = (float*)malloc(max_buf * sizeof(float));
    g_work_b = (float*)malloc(max_buf * sizeof(float));
#endif

    if (!g_work_a || !g_work_b) {
        Serial.println("ERROR: Buffer allocation failed!");
        return false;
    }

    uint32_t total = max_buf * 2 * sizeof(float);
    Serial.printf("Working buffers: 2 x %u floats = %u bytes (%.1f KB)\n",
                  max_buf, total, total / 1024.0f);
    return true;
}

static int forward(const float* input_data) {
    // Copy input into work_a
    memcpy(g_work_a, input_data, g_input_floats * sizeof(float));

    int cur_C = g_input_ch, cur_H = g_input_h, cur_W = g_input_w;

    for (uint32_t li = 0; li < g_num_layers; li++) {
        LayerData& L = g_layers[li];

        if (L.type == 0) {
            // Conv: GroupNorm → TernaryConv → ReLU → MaxPool(2)
            int oH = (cur_H + 2 * (int)L.padding - (int)L.ksize) / (int)L.stride + 1;
            int oW = (cur_W + 2 * (int)L.padding - (int)L.ksize) / (int)L.stride + 1;

            // GroupNorm in-place on g_work_a
            group_norm(g_work_a, cur_C, cur_H, cur_W, L.norm_gamma, L.norm_beta);

            // Conv → g_work_b
            ternary_conv2d(g_work_a, g_work_b, L.weights, L.bias,
                           cur_C, L.out_ch, L.ksize, cur_H, cur_W,
                           L.stride, L.padding);

            // ReLU in-place
            relu(g_work_b, L.out_ch * oH * oW);

            // MaxPool(2) → g_work_a
            int pH = oH / 2, pW = oW / 2;
            max_pool2d(g_work_b, g_work_a, L.out_ch, oH, oW);

            cur_C = L.out_ch; cur_H = pH; cur_W = pW;

        } else {
            // Linear: LayerNorm → TernaryLinear → ReLU (except last)
            int in_f = L.in_features;
            int out_f = L.out_features;

            // LayerNorm in-place on g_work_a
            layer_norm(g_work_a, in_f, L.norm_gamma, L.norm_beta);

            // Linear → g_work_b
            ternary_linear(g_work_a, g_work_b, L.weights, L.bias, in_f, out_f);

            // ReLU except last layer
            bool is_last = (li == g_num_layers - 1);
            if (!is_last) {
                relu(g_work_b, out_f);
            }

            // Swap: copy g_work_b → g_work_a for next layer
            memcpy(g_work_a, g_work_b, out_f * sizeof(float));
        }
    }

    // g_work_a now contains the logits (copied from g_work_b after last linear)
    // But actually last linear output is in g_work_b, then copied to g_work_a
    // Argmax over g_work_a
    int best = 0;
    for (int i = 1; i < g_num_logits; i++) {
        if (g_work_a[i] > g_work_a[best]) best = i;
    }
    return best;
}

// ============================================================
// Arduino entry points
// ============================================================

void setup() {
    Serial.begin(115200);
    while (!Serial) delay(10);

    Serial.println();
    Serial.println("============================================");
    Serial.println("VEDU-BITNET ESP32 INFERENCE ENGINE");
    Serial.println("============================================");

    // Report memory
    Serial.printf("Free heap: %u bytes (%.1f KB)\n",
                  ESP.getFreeHeap(), ESP.getFreeHeap() / 1024.0f);
    Serial.printf("PSRAM size: %u bytes (%.1f KB)\n",
                  ESP.getPsramSize(), ESP.getPsramSize() / 1024.0f);
    Serial.printf("Model in flash: %u bytes (%.1f KB)\n",
                  VEDU_MODEL_SIZE, VEDU_MODEL_SIZE / 1024.0f);

    // Parse model from flash
    if (!parse_model()) {
        Serial.println("FATAL: Model parse failed");
        while (1) delay(1000);
    }

    Serial.printf("Free heap after model: %u bytes\n", ESP.getFreeHeap());

    // Allocate inference buffers
    if (!alloc_buffers()) {
        Serial.println("FATAL: Buffer alloc failed");
        while (1) delay(1000);
    }

    Serial.printf("Free heap after buffers: %u bytes\n", ESP.getFreeHeap());

    // Run test inference
    if (g_has_test) {
        Serial.println();
        Serial.println("--- Running test inference ---");

        // Warmup
        forward(g_test_input);

        // Benchmark
        const int RUNS = 10;
        unsigned long t0 = micros();
        int pred = -1;
        for (int i = 0; i < RUNS; i++) {
            pred = forward(g_test_input);
        }
        unsigned long t1 = micros();

        float avg_us = (float)(t1 - t0) / RUNS;

        Serial.println();
        Serial.println("========== RESULTS ==========");
        Serial.printf("  Prediction: %d\n", pred);
        Serial.printf("  Expected:   %d\n", g_test_expected);
        Serial.printf("  Match:      %s\n",
                      pred == g_test_expected ? "YES" : "NO");
        Serial.printf("  Avg time:   %.0f us (%.1f ms)\n", avg_us, avg_us/1000.0f);
        Serial.printf("  Throughput: %.0f inf/sec\n", 1e6f / avg_us);
        Serial.println("=============================");

        // Print logits
        Serial.print("  Logits: [");
        for (int i = 0; i < g_num_logits; i++) {
            Serial.printf("%.2f%s", g_work_a[i], i < g_num_logits - 1 ? ", " : "");
        }
        Serial.println("]");
    }

    Serial.println();
    Serial.println("Ready. Send 'r' over Serial to re-run inference.");
}

void loop() {
    if (Serial.available()) {
        char c = Serial.read();
        if (c == 'r' || c == 'R') {
            if (g_has_test) {
                unsigned long t0 = micros();
                int pred = forward(g_test_input);
                unsigned long t1 = micros();
                Serial.printf("Prediction: %d (%.1f ms)\n",
                              pred, (t1 - t0) / 1000.0f);
            }
        }
    }
    delay(10);
}
