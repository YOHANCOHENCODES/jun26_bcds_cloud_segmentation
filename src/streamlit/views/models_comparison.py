import pandas as pd

import streamlit as st


st.write('## Models Comparison')

st.markdown("""
Four model configurations were developed and evaluated in this project: a
compact baseline U-Net, a U-Net with a ResNet34 encoder, a U-Net with a
ResNet50 encoder, and a PSPNet with a ResNet34 encoder. All four were
evaluated on the same held-out 555-image validation split.
""")


st.write('### Final Validation Comparison')

final_comparison = pd.DataFrame({
    'Model': ['U-Net', 'U-Net', 'PSPNet'],
    'Encoder': ['ResNet34', 'ResNet50', 'ResNet34'],
    'Best Validation Macro Dice': [0.5974, 0.6118, 0.6114],
})

st.dataframe(
    final_comparison,
    hide_index=True,
    use_container_width=True,
)


st.write('### Per-Class Comparison: U-Net (ResNet50) vs. PSPNet (ResNet34)')

per_class = pd.DataFrame({
    'Cloud type': ['Fish', 'Flower', 'Gravel', 'Sugar', 'Mean (macro)'],
    'PSPNet (ResNet34) Dice': [0.5125, 0.6947, 0.5829, 0.6253, 0.6114],
    'U-Net (ResNet50) Dice': [0.5478, 0.6962, 0.5729, 0.6301, 0.6118],
})

st.dataframe(
    per_class,
    hide_index=True,
    use_container_width=True,
)


st.markdown("""
**Selected model:** U-Net with ResNet50 was selected as the final model
because it achieved the highest Macro Dice score among the evaluated
configurations (0.6118). Its performance was very close to PSPNet with
ResNet34 (0.6114), indicating highly similar aggregate validation performance.

Across the four cloud categories, **Flower** achieved the highest Dice score
for both final models, while **Fish** had the lowest Dice score. The largest
difference between the two final models was observed for Fish (0.0353 Dice).
""")