import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

# Set style for better visibility
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams.update({'font.size': 14})

# Setup
vocab = {"hello": 0, "world": 1, "cat": 2, "dog": 3}
vocab_words = list(vocab.keys())
vocab_size = 4

# Create model
torch.manual_seed(42)
model = torch.nn.Linear(vocab_size, vocab_size)

print("🚀 CLEAR SFT TRAINING VISUALIZATIONS")
print("=" * 50)

# Create outputs directory if it doesn't exist
os.makedirs('outputs', exist_ok=True)

# Prepare data
input_token = torch.tensor([0])
input_onehot = F.one_hot(input_token, vocab_size).float()
logits = model(input_onehot)
probs = F.softmax(logits, dim=1)
target = torch.tensor([1])
loss = F.cross_entropy(logits, target)
loss.backward()

# Extract values for visualization
input_viz = input_onehot.numpy().flatten()
logits_viz = logits.detach().numpy().flatten()
probs_viz = probs.detach().numpy().flatten()
gradients = model.weight.grad.numpy()
grad_for_input = gradients[:, 0]

print(f"Target: 'world' (we want this to have high probability)")
print(f"Current 'world' probability: {probs_viz[1]:.3f}")
print(f"Loss: {loss.item():.3f}")

# ============================================================================
# VISUALIZATION 1: INPUT AND TARGET
# ============================================================================
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
colors = ['red' if x > 0 else 'lightgray' for x in input_viz]
bars = plt.bar(vocab_words, input_viz, color=colors, edgecolor='black', linewidth=2)
plt.title('🎯 INPUT: "hello"', fontsize=18, fontweight='bold', pad=20)
plt.ylabel('One-Hot Value', fontsize=14)
plt.ylim(0, 1.2)

for i, v in enumerate(input_viz):
    plt.text(i, v + 0.05, f'{v:.1f}', ha='center', fontweight='bold', fontsize=16)

plt.text(0, 0.5, '← ACTIVE', ha='center', fontweight='bold', color='red', fontsize=12)

plt.subplot(1, 2, 2)
target_viz = [0, 1, 0, 0]  # What we want
colors = ['lightgray' if x == 0 else 'lime' for x in target_viz]
bars = plt.bar(vocab_words, target_viz, color=colors, edgecolor='black', linewidth=2)
plt.title('🎯 TARGET: "world"', fontsize=18, fontweight='bold', pad=20)
plt.ylabel('Target Value', fontsize=14)
plt.ylim(0, 1.2)

for i, v in enumerate(target_viz):
    plt.text(i, v + 0.05, f'{v:.1f}', ha='center', fontweight='bold', fontsize=16)

plt.text(1, 0.5, '← WANT THIS!', ha='center', fontweight='bold', color='lime', fontsize=12)

plt.tight_layout()
plt.suptitle('STEP 1: What Goes In vs What We Want Out', fontsize=20, fontweight='bold', y=1.02)
plt.savefig('outputs/01_input_vs_target.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# VISUALIZATION 2: MODEL PROCESSING
# ============================================================================
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
bars = plt.bar(vocab_words, logits_viz, color='orange', edgecolor='black', linewidth=2)
plt.title('⚡ RAW LOGITS', fontsize=16, fontweight='bold')
plt.ylabel('Logit Value', fontsize=14)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)

for i, v in enumerate(logits_viz):
    plt.text(i, v + 0.05, f'{v:.2f}', ha='center', fontweight='bold', fontsize=12)

plt.subplot(1, 3, 2)
plt.arrow(0.2, 0.5, 0.6, 0, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
plt.text(0.5, 0.7, 'SOFTMAX', ha='center', fontweight='bold', fontsize=16, color='blue')
plt.text(0.5, 0.3, 'Converts to\nProbabilities', ha='center', fontsize=12)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.axis('off')
plt.title('🔄 CONVERSION', fontsize=16, fontweight='bold')

plt.subplot(1, 3, 3)
colors = ['green' if i == 1 else 'red' for i in range(4)]
bars = plt.bar(vocab_words, probs_viz, color=colors, edgecolor='black', linewidth=2)
plt.title('📊 PROBABILITIES', fontsize=16, fontweight='bold')
plt.ylabel('Probability', fontsize=14)
plt.ylim(0, 0.5)

for i, v in enumerate(probs_viz):
    plt.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold', fontsize=12)

# Highlight the problem
plt.text(1, probs_viz[1] + 0.08, f'TOO LOW!\nShould be 1.0', ha='center',
         fontweight='bold', color='red', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.suptitle('STEP 2: How Model Processes Input', fontsize=20, fontweight='bold', y=1.02)
plt.savefig('outputs/02_model_processing.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# VISUALIZATION 3: THE MAGIC GRADIENTS (HEATMAPS)
# ============================================================================
plt.figure(figsize=(15, 10))

# Full gradient matrix heatmap
plt.subplot(2, 2, 1)
grad_df = pd.DataFrame(gradients, index=vocab_words, columns=vocab_words)
sns.heatmap(grad_df, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            cbar_kws={'label': 'Gradient Value'}, linewidths=1, linecolor='black')
plt.title('🎯 FULL GRADIENT MATRIX', fontsize=16, fontweight='bold')
plt.xlabel('Input Word (columns)', fontsize=12)
plt.ylabel('Output Word (rows)', fontsize=12)

# Add explanation
plt.text(4.2, 2, 'ONLY FIRST COLUMN\nHAS VALUES\n(input was "hello")',
         fontsize=11, fontweight='bold', color='blue',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Focus on the active gradients (first column only)
plt.subplot(2, 2, 2)
grad_active = gradients[:, 0].reshape(-1, 1)  # Make it 2D for heatmap
grad_active_df = pd.DataFrame(grad_active, index=vocab_words, columns=['Gradient'])
sns.heatmap(grad_active_df, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            cbar_kws={'label': 'Gradient Value'}, linewidths=2, linecolor='black')
plt.title('🔥 ACTIVE GRADIENTS\n(for input "hello")', fontsize=16, fontweight='bold')
plt.xlabel('')

# Add arrows and explanations
for i, val in enumerate(grad_for_input):
    if val < 0:  # Negative (increase probability)
        plt.annotate('INCREASE ⬆️', xy=(0.5, i + 0.5), xytext=(1.5, i + 0.5),
                     ha='left', va='center', fontweight='bold', color='green', fontsize=12,
                     arrowprops=dict(arrowstyle='->', color='green', lw=3))
    else:  # Positive (decrease probability)
        plt.annotate('DECREASE ⬇️', xy=(0.5, i + 0.5), xytext=(1.5, i + 0.5),
                     ha='left', va='center', fontweight='bold', color='red', fontsize=12,
                     arrowprops=dict(arrowstyle='->', color='red', lw=3))

# Probability comparison heatmap
plt.subplot(2, 2, 3)
prob_comparison = np.array([probs_viz, [0, 1, 0, 0]])  # Current vs Target
prob_df = pd.DataFrame(prob_comparison.T, index=vocab_words, columns=['Current', 'Target'])
sns.heatmap(prob_df, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1,
            cbar_kws={'label': 'Probability'}, linewidths=2, linecolor='black')
plt.title('📊 CURRENT vs TARGET\nProbabilities', fontsize=16, fontweight='bold')

# Gradient direction visualization
plt.subplot(2, 2, 4)
# Create a visual representation of gradient directions
direction_matrix = np.zeros((4, 3))
for i, grad_val in enumerate(grad_for_input):
    if grad_val < 0:  # Increase
        direction_matrix[i, 2] = abs(grad_val)  # Green column
    else:  # Decrease
        direction_matrix[i, 0] = grad_val  # Red column

direction_df = pd.DataFrame(direction_matrix, index=vocab_words,
                            columns=['Decrease', 'Neutral', 'Increase'])
sns.heatmap(direction_df, annot=False, cmap='RdYlGn', cbar_kws={'label': 'Magnitude'},
            linewidths=2, linecolor='black')
plt.title('🎯 GRADIENT DIRECTIONS\n& MAGNITUDES', fontsize=16, fontweight='bold')

# Add text annotations
for i, grad_val in enumerate(grad_for_input):
    if grad_val < 0:
        plt.text(2, i + 0.5, f'↑ {abs(grad_val):.3f}', ha='center', va='center',
                 fontweight='bold', color='darkgreen', fontsize=12)
    else:
        plt.text(0, i + 0.5, f'↓ {grad_val:.3f}', ha='center', va='center',
                 fontweight='bold', color='darkred', fontsize=12)

plt.tight_layout()
plt.savefig('outputs/03_gradients_heatmaps.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# BONUS: GRADIENT FORMULA BREAKDOWN HEATMAP
# ============================================================================
plt.figure(figsize=(12, 8))

# Show step-by-step formula calculation
formula_steps = {
    'Current Prob': probs_viz,
    'Target Prob': [0, 1, 0, 0],
    'Difference': probs_viz - np.array([0, 1, 0, 0]),
    'Input': input_viz,
    'Gradient': grad_for_input
}

formula_df = pd.DataFrame(formula_steps, index=vocab_words)

plt.subplot(1, 2, 1)
sns.heatmap(formula_df, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            cbar_kws={'label': 'Value'}, linewidths=1, linecolor='black')
plt.title('🔬 GRADIENT FORMULA BREAKDOWN', fontsize=16, fontweight='bold')
plt.xlabel('Calculation Steps', fontsize=12)

# Show the actual formula
plt.subplot(1, 2, 2)
plt.axis('off')
formula_text = f"""
🔬 GRADIENT CALCULATION

For each output word:

gradient = (current_prob - target_prob) × input

EXAMPLES:

'world' (target):
  = ({probs_viz[1]:.3f} - 1.0) × {input_viz[0]:.1f}
  = {grad_for_input[1]:.3f} ← NEGATIVE = INCREASE!

'hello' (wrong):
  = ({probs_viz[0]:.3f} - 0.0) × {input_viz[0]:.1f}  
  = {grad_for_input[0]:.3f} ← POSITIVE = DECREASE!

🧠 KEY INSIGHT:
• Negative gradient → Increase probability
• Positive gradient → Decrease probability
• Math automatically finds right direction!
"""

plt.text(0.05, 0.95, formula_text, transform=plt.gca().transAxes, fontsize=13,
         verticalalignment='top', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, pad=1))

plt.tight_layout()
plt.savefig('outputs/04_gradient_formula.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# VISUALIZATION 4: WEIGHT UPDATES
# ============================================================================
learning_rate = 0.1
old_weights = model.weight.data.clone()
with torch.no_grad():
    model.weight -= learning_rate * model.weight.grad

weight_changes = (model.weight.data - old_weights).numpy()
weight_change_for_input = weight_changes[:, 0]

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
colors = ['red' if x < 0 else 'green' for x in weight_change_for_input]
bars = plt.bar(vocab_words, weight_change_for_input, color=colors, edgecolor='black', linewidth=2)
plt.title('🔄 WEIGHT CHANGES', fontsize=18, fontweight='bold')
plt.ylabel('Weight Change', fontsize=14)
plt.axhline(y=0, color='black', linestyle='-', linewidth=2)

for i, v in enumerate(weight_change_for_input):
    plt.text(i, v + (0.008 if v > 0 else -0.015), f'{v:.3f}', ha='center', fontweight='bold', fontsize=12)

# Show the update formula
plt.text(1, weight_change_for_input[1] + 0.05, f'= -lr × gradient\n= -0.1 × {grad_for_input[1]:.3f}',
         ha='center', fontweight='bold', color='green', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

plt.subplot(1, 2, 2)
# Show direction of change
directions = ['⬇️' if x < 0 else '⬆️' for x in weight_change_for_input]
colors = ['red' if x < 0 else 'green' for x in weight_change_for_input]

for i, (direction, color) in enumerate(zip(directions, colors)):
    plt.text(i, 0.5, direction, ha='center', fontsize=40, color=color)
    if i == 1:
        plt.text(i, 0.2, 'WEIGHT UP\n→ HIGHER PROB', ha='center', fontweight='bold', color='green')
    else:
        plt.text(i, 0.2, 'WEIGHT DOWN\n→ LOWER PROB', ha='center', fontweight='bold', color='red')

plt.xlim(-0.5, 3.5)
plt.ylim(0, 1)
plt.title('📈 UPDATE DIRECTIONS', fontsize=18, fontweight='bold')
plt.xticks(range(4), vocab_words, fontsize=14)
plt.yticks([])

plt.tight_layout()
plt.savefig('outputs/05_weight_updates.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# VISUALIZATION 5: BEFORE VS AFTER
# ============================================================================
model.zero_grad()
new_logits = model(input_onehot)
new_probs = F.softmax(new_logits, dim=1)
new_probs_viz = new_probs.detach().numpy().flatten()

plt.figure(figsize=(12, 8))

# Before vs After comparison
x = np.arange(len(vocab_words))
width = 0.35

plt.subplot(2, 1, 1)
bars_old = plt.bar(x - width / 2, probs_viz, width, label='BEFORE', color='red', alpha=0.8, edgecolor='black')
bars_new = plt.bar(x + width / 2, new_probs_viz, width, label='AFTER', color='green', alpha=0.8, edgecolor='black')

plt.title('🎉 TRAINING RESULT: Before vs After', fontsize=18, fontweight='bold', pad=20)
plt.ylabel('Probability', fontsize=14)
plt.xticks(x, vocab_words, fontsize=14)
plt.legend(fontsize=14)
plt.ylim(0, 0.4)

# Add value labels
for i, (old, new) in enumerate(zip(probs_viz, new_probs_viz)):
    plt.text(i - width / 2, old + 0.01, f'{old:.3f}', ha='center', fontweight='bold', fontsize=11)
    plt.text(i + width / 2, new + 0.01, f'{new:.3f}', ha='center', fontweight='bold', fontsize=11)

    # Show improvement for target
    if i == 1:  # world
        change = new - old
        plt.annotate(f'📈 IMPROVED!\n+{change:.3f}',
                     xy=(i + width / 2, new + 0.05),
                     ha='center', fontweight='bold', color='green', fontsize=12,
                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

# Summary metrics
plt.subplot(2, 1, 2)
plt.axis('off')

summary_text = f"""
🎯 TRAINING SUMMARY

✅ SUCCESS! The model learned:

📊 'WORLD' PROBABILITY:
   Before: {probs_viz[1]:.3f}
   After:  {new_probs_viz[1]:.3f}
   Change: +{new_probs_viz[1] - probs_viz[1]:.3f}

🔄 HOW IT WORKED:
   • Negative gradient → Increased target probability
   • Positive gradients → Decreased wrong probabilities
   • Math automatically found the right direction!

🧠 KEY INSIGHT:
   ∇θ log πθ = Smart learning signal that:
   • Knows which weights to increase/decrease
   • Knows how much to change each weight
   • Does all the thinking for us!
"""

plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, fontsize=14,
         verticalalignment='top', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, pad=1))

plt.tight_layout()
plt.savefig('outputs/06_before_vs_after.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\n🧠 FINAL INSIGHT:")
print(f"The gradient ∇θ log πθ is like a smart GPS for learning:")
print(f"• It automatically knows which direction to go")
print(f"• It knows how big steps to take")
print(f"• It handles all the complex math for us!")
print(f"")
print(f"This is why SFT works so well - the math does the thinking! 🤯")