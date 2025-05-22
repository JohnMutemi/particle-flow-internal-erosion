import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

fig, ax = plt.subplots(figsize=(8, 5))

# Draw boxes
ax.text(0.1, 0.5, 'DEM\n(Particles)', ha='center', va='center', fontsize=14, bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', ec='navy'))
ax.text(0.9, 0.5, 'CFD\n(Fluid)', ha='center', va='center', fontsize=14, bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', ec='darkgreen'))
ax.text(0.5, 0.8, 'Coupling\nManager', ha='center', va='center', fontsize=13, bbox=dict(boxstyle='round,pad=0.4', fc='wheat', ec='saddlebrown'))

# Draw arrows
arrow1 = FancyArrowPatch((0.22, 0.55), (0.45, 0.75), arrowstyle='->', mutation_scale=20, color='gray', lw=2)
arrow2 = FancyArrowPatch((0.78, 0.55), (0.55, 0.75), arrowstyle='->', mutation_scale=20, color='gray', lw=2)
arrow3 = FancyArrowPatch((0.5, 0.75), (0.22, 0.55), arrowstyle='->', mutation_scale=20, color='gray', lw=2, linestyle='dashed')
arrow4 = FancyArrowPatch((0.5, 0.75), (0.78, 0.55), arrowstyle='->', mutation_scale=20, color='gray', lw=2, linestyle='dashed')
ax.add_patch(arrow1)
ax.add_patch(arrow2)
ax.add_patch(arrow3)
ax.add_patch(arrow4)

# Add labels for arrows
ax.text(0.33, 0.68, 'Forces', fontsize=11, color='black', ha='center')
ax.text(0.67, 0.68, 'Velocities', fontsize=11, color='black', ha='center')
ax.text(0.33, 0.58, 'Drag/Lift', fontsize=10, color='gray', ha='center')
ax.text(0.67, 0.58, 'Feedback', fontsize=10, color='gray', ha='center')

ax.axis('off')
plt.title('CFD-DEM (Flow-Solid) Coupling Framework', fontsize=15, pad=20)
plt.tight_layout()
plt.savefig('results/coupling_framework.png')
plt.close() 