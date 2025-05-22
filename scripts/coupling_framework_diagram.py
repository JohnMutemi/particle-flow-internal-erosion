import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.font_manager as fm
import matplotlib

def create_coupling_diagram(lang_code='en'):
    """Create the coupling framework diagram with bilingual labels."""
    # Set font for Chinese characters
    if lang_code == 'zh':
        # Explicitly set to Noto Sans CJK JP (available on system)
        plt.rcParams['font.family'] = 'Noto Sans CJK JP'
    
    # Language dictionary for labels
    labels = {
        'en': {
            'dem': 'DEM\n(Particles)',
            'cfd': 'CFD\n(Fluid)',
            'coupling': 'Coupling\nManager',
            'forces': 'Forces',
            'velocities': 'Velocities',
            'drag_lift': 'Drag/Lift',
            'feedback': 'Feedback',
            'title': 'CFD-DEM (Flow-Solid) Coupling Framework'
        },
        'zh': {
            'dem': 'DEM\n(颗粒)',
            'cfd': 'CFD\n(流体)',
            'coupling': '耦合\n管理器',
            'forces': '力',
            'velocities': '速度',
            'drag_lift': '阻力/升力',
            'feedback': '反馈',
            'title': 'CFD-DEM (流固) 耦合框架'
        }
    }
    
    # Get labels for current language
    lang = labels[lang_code]
    
    fig, ax = plt.subplots(figsize=(8, 5))

    # Draw boxes
    ax.text(0.1, 0.5, lang['dem'], ha='center', va='center', fontsize=14, 
            bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', ec='navy'))
    ax.text(0.9, 0.5, lang['cfd'], ha='center', va='center', fontsize=14, 
            bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', ec='darkgreen'))
    ax.text(0.5, 0.8, lang['coupling'], ha='center', va='center', fontsize=13, 
            bbox=dict(boxstyle='round,pad=0.4', fc='wheat', ec='saddlebrown'))

    # Draw arrows
    arrow1 = FancyArrowPatch((0.22, 0.55), (0.45, 0.75), arrowstyle='->', 
                            mutation_scale=20, color='gray', lw=2)
    arrow2 = FancyArrowPatch((0.78, 0.55), (0.55, 0.75), arrowstyle='->', 
                            mutation_scale=20, color='gray', lw=2)
    arrow3 = FancyArrowPatch((0.5, 0.75), (0.22, 0.55), arrowstyle='->', 
                            mutation_scale=20, color='gray', lw=2, linestyle='dashed')
    arrow4 = FancyArrowPatch((0.5, 0.75), (0.78, 0.55), arrowstyle='->', 
                            mutation_scale=20, color='gray', lw=2, linestyle='dashed')
    ax.add_patch(arrow1)
    ax.add_patch(arrow2)
    ax.add_patch(arrow3)
    ax.add_patch(arrow4)

    # Add labels for arrows
    ax.text(0.33, 0.68, lang['forces'], fontsize=11, color='black', ha='center')
    ax.text(0.67, 0.68, lang['velocities'], fontsize=11, color='black', ha='center')
    ax.text(0.33, 0.58, lang['drag_lift'], fontsize=10, color='gray', ha='center')
    ax.text(0.67, 0.58, lang['feedback'], fontsize=10, color='gray', ha='center')

    ax.axis('off')
    plt.title(lang['title'], fontsize=15, pad=20)
    plt.tight_layout()
    
    # Save both English and Chinese versions
    plt.savefig(f'results/coupling_framework_{lang_code}.png')
    plt.close()

if __name__ == "__main__":
    # Create both English and Chinese versions
    create_coupling_diagram('en')
    create_coupling_diagram('zh') 