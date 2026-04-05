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
 * Memory budget (ESP32 has 520KB SRAM):
 *   Model weights:    0 KB (stored in flash, not RAM)
 *   Working buffers: ~145 KB (stack + heap for inference)
 *   Free for app:   ~375 KB
 *
 * Board: ESP32 (any variant) or ESP32-S3
 * Upload: Arduino IDE → Tools → Board → ESP32 Dev Module
 *
 * For real camera deployment, add ESP32-CAM + OV2640
 * and pipe the downscaled grayscale frame into forward().
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
        g_test_input = (float*)malloc(inp_size * 4);
        r.read_f32_array(g_test_input, inp_size);
        r.read_i32(g_test_expected);
        // Skip logits (we don't need them on ESP32)
        Serial.printf("Test vector loaded (expected: %d)\n", g_test_expected);
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
// Full forward pass
// ============================================================
// Allocate working buffers on heap (too large for ESP32 stack)

static float* buf_a = nullptr;  // [1*28*28] = 3136 bytes
static float* buf_b = nullptr;  // [16*28*28] = 50176 bytes
static float* buf_c = nullptr;  // [16*14*14] = 12544 bytes
static float* buf_d = nullptr;  // [16*14*14] = 12544 bytes
static float* buf_e = nullptr;  // [32*14*14] = 25088 bytes
static float* buf_f = nullptr;  // [32*7*7]   = 6272 bytes
static float* buf_g = nullptr;  // [1568]     = 6272 bytes
static float* buf_h = nullptr;  // [128]      = 512 bytes
static float* buf_i = nullptr;  // [128]      = 512 bytes
static float* buf_j = nullptr;  // [10]       = 40 bytes
// Total buffers: ~117 KB

static bool alloc_buffers() {
    // Use PSRAM (ps_malloc) for large buffers — internal SRAM is only 520KB
    // and gets tight after model params are loaded. The board has 2MB PSRAM.
#ifdef ESP32
    #define ALLOC(sz) (float*)ps_malloc(sz)
#else
    #define ALLOC(sz) (float*)malloc(sz)
#endif
    buf_a = ALLOC(1 * 28 * 28 * sizeof(float));
    buf_b = ALLOC(16 * 28 * 28 * sizeof(float));
    buf_c = ALLOC(16 * 14 * 14 * sizeof(float));
    buf_d = ALLOC(16 * 14 * 14 * sizeof(float));
    buf_e = ALLOC(32 * 14 * 14 * sizeof(float));
    buf_f = ALLOC(32 * 7 * 7 * sizeof(float));
    buf_g = ALLOC(1568 * sizeof(float));
    buf_h = ALLOC(128 * sizeof(float));
    buf_i = ALLOC(128 * sizeof(float));
    buf_j = ALLOC(10 * sizeof(float));
#undef ALLOC

    if (!buf_a || !buf_b || !buf_c || !buf_d || !buf_e ||
        !buf_f || !buf_g || !buf_h || !buf_i || !buf_j) {
        Serial.println("ERROR: Buffer allocation failed!");
        return false;
    }

    uint32_t total = (1*28*28 + 16*28*28 + 16*14*14 + 16*14*14 +
                      32*14*14 + 32*7*7 + 1568 + 128 + 128 + 10) * 4;
    Serial.printf("Working buffers: %u bytes (%.1f KB)\n", total, total/1024.0f);
    return true;
}

static int forward(const float* input_784) {
    // L0: GroupNorm → Conv1(1→16, 3x3, pad=1) → ReLU → MaxPool(2)
    memcpy(buf_a, input_784, 784 * sizeof(float));
    group_norm(buf_a, 1, 28, 28, g_layers[0].norm_gamma, g_layers[0].norm_beta);
    ternary_conv2d(buf_a, buf_b, g_layers[0].weights, g_layers[0].bias,
                   1, 16, 3, 28, 28, 1, 1);
    relu(buf_b, 16 * 28 * 28);
    max_pool2d(buf_b, buf_c, 16, 28, 28);

    // L1: GroupNorm → Conv2(16→32, 3x3, pad=1) → ReLU → MaxPool(2)
    memcpy(buf_d, buf_c, 16 * 14 * 14 * sizeof(float));
    group_norm(buf_d, 16, 14, 14, g_layers[1].norm_gamma, g_layers[1].norm_beta);
    ternary_conv2d(buf_d, buf_e, g_layers[1].weights, g_layers[1].bias,
                   16, 32, 3, 14, 14, 1, 1);
    relu(buf_e, 32 * 14 * 14);
    max_pool2d(buf_e, buf_f, 32, 14, 14);

    // Flatten: [32,7,7] → [1568]

    // L2: LayerNorm → FC1(1568→128) → ReLU
    memcpy(buf_g, buf_f, 1568 * sizeof(float));
    layer_norm(buf_g, 1568, g_layers[2].norm_gamma, g_layers[2].norm_beta);
    ternary_linear(buf_g, buf_h, g_layers[2].weights, g_layers[2].bias,
                   1568, 128);
    relu(buf_h, 128);

    // L3: LayerNorm → FC2(128→10)
    memcpy(buf_i, buf_h, 128 * sizeof(float));
    layer_norm(buf_i, 128, g_layers[3].norm_gamma, g_layers[3].norm_beta);
    ternary_linear(buf_i, buf_j, g_layers[3].weights, g_layers[3].bias,
                   128, 10);

    // Argmax
    int best = 0;
    for (int i = 1; i < 10; i++) {
        if (buf_j[i] > buf_j[best]) best = i;
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
        for (int i = 0; i < 10; i++) {
            Serial.printf("%.2f%s", buf_j[i], i < 9 ? ", " : "");
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
