"""
A TINY NEURAL NETWORK FROM SCRATCH
===================================
Goal: Teach a model to learn the AND logic gate

AND gate truth table:
  Input1  Input2  →  Expected Output
    0       0     →      0
    0       1     →      0
    1       0     →      0
    1       1     →      1

The model starts with RANDOM weights and learns the correct answer
by adjusting weights step by step.
"""

import random
import math

# ============================================================
# STEP 1: THE DATA (what we want the model to learn)
# ============================================================
# Each entry: [input1, input2, expected_output]
training_data = [
    [0, 0, 0],   # 0 AND 0 = 0
    [0, 1, 0],   # 0 AND 1 = 0
    [1, 0, 0],   # 1 AND 0 = 0
    [1, 1, 1],   # 1 AND 1 = 1
]

# ============================================================
# STEP 2: THE WEIGHTS (start random — the model knows nothing)
# ============================================================
random.seed(42)  # so you get the same results as me

# Our tiny model:
#
#   Input1 ──(w1)──┐
#                   ├──► [add + bias + activation] ──► Output
#   Input2 ──(w2)──┘
#                   │
#           bias ───┘
#
# Bias is like a weight that's always connected to a "1"
# It lets the model shift its answer up or down

w1 = random.uniform(-1, 1)    # weight for input 1
w2 = random.uniform(-1, 1)    # weight for input 2
bias = random.uniform(-1, 1)  # the shift knob

print("=" * 60)
print("STARTING WEIGHTS (random — the model is clueless)")
print("=" * 60)
print(f"  w1   = {w1:.4f}")
print(f"  w2   = {w2:.4f}")
print(f"  bias = {bias:.4f}")

# ============================================================
# STEP 3: THE ACTIVATION FUNCTION (sigmoid)
# ============================================================
# Problem: Input × Weight can give any number (-∞ to +∞)
# We need the output to be between 0 and 1 (like a probability)
#
# Sigmoid squishes any number into the 0-1 range:
#   Big positive number → close to 1
#   Big negative number → close to 0
#   Zero                → exactly 0.5

def sigmoid(x):
    """Squish any number into range 0 to 1"""
    return 1 / (1 + math.exp(-x))

print("\nSigmoid examples:")
print(f"  sigmoid(-5)  = {sigmoid(-5):.4f}  ← big negative → near 0")
print(f"  sigmoid( 0)  = {sigmoid(0):.4f}  ← zero → exactly 0.5")
print(f"  sigmoid( 5)  = {sigmoid(5):.4f}  ← big positive → near 1")

# ============================================================
# STEP 4: FORWARD PASS (the model makes a prediction)
# ============================================================
# This is the core math:
#   1. Multiply each input by its weight
#   2. Add them all together (plus bias)
#   3. Squish through sigmoid

def predict(input1, input2):
    """The model's prediction — this is ALL a neural net does"""
    # Step A: Weighted sum
    weighted_sum = (input1 * w1) + (input2 * w2) + bias

    # Step B: Squish to 0-1
    output = sigmoid(weighted_sum)

    return output

print("\n" + "=" * 60)
print("BEFORE TRAINING — Model's guesses (should be terrible)")
print("=" * 60)
for data in training_data:
    input1, input2, expected = data
    guess = predict(input1, input2)
    print(f"  Input: ({input1}, {input2})  Expected: {expected}  Got: {guess:.4f}  {'✓' if round(guess) == expected else '✗ WRONG'}")

# ============================================================
# STEP 5: TRAINING — The model learns!
# ============================================================
# How training works:
#   1. Model makes a guess
#   2. Calculate error (how wrong was it?)
#   3. Adjust weights to be less wrong
#   4. Repeat thousands of times
#
# The "learning rate" controls how big each adjustment is:
#   Too big → model overshoots and never learns
#   Too small → learning takes forever
#   Just right → model gradually converges

learning_rate = 0.5
epochs = 10000  # how many times to loop through all data

print("\n" + "=" * 60)
print("TRAINING — Watch the model learn!")
print("=" * 60)

for epoch in range(epochs):
    total_error = 0

    for data in training_data:
        input1, input2, expected = data

        # --- FORWARD PASS (predict) ---
        weighted_sum = (input1 * w1) + (input2 * w2) + bias
        output = sigmoid(weighted_sum)

        # --- CALCULATE ERROR ---
        error = expected - output  # positive = too low, negative = too high
        total_error += error ** 2  # square it so negatives don't cancel out

        # --- BACKWARD PASS (learn) ---
        # How much should each weight change?
        # This uses calculus (chain rule) but the result is simple:
        #
        #   adjustment = error × slope_of_sigmoid × input
        #
        # slope_of_sigmoid = output × (1 - output)
        # This means: if the model is very confident (output near 0 or 1),
        # make small changes. If unsure (output near 0.5), make bigger changes.

        slope = output * (1 - output)

        # Each weight's adjustment:
        w1_adjustment = error * slope * input1
        w2_adjustment = error * slope * input2
        bias_adjustment = error * slope * 1  # bias input is always 1

        # --- UPDATE WEIGHTS ---
        w1 += learning_rate * w1_adjustment
        w2 += learning_rate * w2_adjustment
        bias += learning_rate * bias_adjustment

    # Print progress every 1000 epochs
    if epoch % 2000 == 0 or epoch == epochs - 1:
        print(f"\n  Epoch {epoch:>5}/{epochs}  |  Total Error: {total_error:.6f}")
        print(f"  Weights: w1={w1:.4f}  w2={w2:.4f}  bias={bias:.4f}")
        for data in training_data:
            input1, input2, expected = data
            weighted_sum = (input1 * w1) + (input2 * w2) + bias
            output = sigmoid(weighted_sum)
            print(f"    ({input1},{input2}) → {output:.4f}  (want {expected})  {'✓' if round(output) == expected else '✗'}")

# ============================================================
# STEP 6: FINAL RESULTS
# ============================================================
print("\n" + "=" * 60)
print("AFTER TRAINING — The model learned!")
print("=" * 60)
print(f"  Final weights:")
print(f"    w1   = {w1:.4f}  (how much input1 matters)")
print(f"    w2   = {w2:.4f}  (how much input2 matters)")
print(f"    bias = {bias:.4f}  (the shift)")
print()

for data in training_data:
    input1, input2, expected = data
    weighted_sum = (input1 * w1) + (input2 * w2) + bias
    output = sigmoid(weighted_sum)
    print(f"  Input: ({input1}, {input2})  Expected: {expected}  Got: {output:.4f}  {'✓ CORRECT' if round(output) == expected else '✗ WRONG'}")

# ============================================================
# STEP 7: LET'S TRACE ONE FULL CALCULATION
# ============================================================
print("\n" + "=" * 60)
print("FULL MATH TRACE — Input (1, 1) should give ~1")
print("=" * 60)
i1, i2 = 1, 1
print(f"""
  Step 1: Multiply inputs by weights
    Input1 × w1 = {i1} × {w1:.4f} = {i1 * w1:.4f}
    Input2 × w2 = {i2} × {w2:.4f} = {i2 * w2:.4f}

  Step 2: Add everything together (including bias)
    sum = {i1 * w1:.4f} + {i2 * w2:.4f} + {bias:.4f} = {i1*w1 + i2*w2 + bias:.4f}

  Step 3: Squish through sigmoid
    sigmoid({i1*w1 + i2*w2 + bias:.4f}) = 1 / (1 + e^(-{i1*w1 + i2*w2 + bias:.4f}))
                                        = {sigmoid(i1*w1 + i2*w2 + bias):.4f}

  That's close to 1 → model says YES! ✓
""")

print("=" * 60)
print("FULL MATH TRACE — Input (0, 1) should give ~0")
print("=" * 60)
i1, i2 = 0, 1
print(f"""
  Step 1: Multiply inputs by weights
    Input1 × w1 = {i1} × {w1:.4f} = {i1 * w1:.4f}
    Input2 × w2 = {i2} × {w2:.4f} = {i2 * w2:.4f}

  Step 2: Add everything together (including bias)
    sum = {i1 * w1:.4f} + {i2 * w2:.4f} + {bias:.4f} = {i1*w1 + i2*w2 + bias:.4f}

  Step 3: Squish through sigmoid
    sigmoid({i1*w1 + i2*w2 + bias:.4f}) = 1 / (1 + e^(-{i1*w1 + i2*w2 + bias:.4f}))
                                        = {sigmoid(i1*w1 + i2*w2 + bias):.4f}

  That's close to 0 → model says NO! ✓
""")

print("THE ENTIRE MODEL IS JUST 3 NUMBERS: w1, w2, and bias.")
print("Everything else is multiply, add, and sigmoid.")
print("GPT is the same thing... just with 8,000,000,000 numbers instead of 3.")
