import pandas as pd

def apply_clinical_labels():
    # Load the merged data from Step 2
    df = pd.read_csv('data/processed_nutrition.csv')

    # 1. Label for Iron Deficiency (LBDFERSI)
    # 1 = Deficient, 0 = Normal
    df['iron_deficient'] = (df['LBDFERSI'] < 15).astype(int)

    # 2. Label for Vitamin D Deficiency (LBXVIDMS)
    # NHANES often reports in nmol/L. 20 ng/mL is roughly 50 nmol/L.
    df['vit_d_deficient'] = (df['LBXVIDMS'] < 50).astype(int)

    # 3. Create a 'General Deficiency' label 
    # (If they are deficient in either, we mark them as 1)
    df['any_deficiency'] = ((df['iron_deficient'] == 1) | (df['vit_d_deficient'] == 1)).astype(int)

    # Save the labeled dataset
    df.to_csv('data/labeled_nutrition.csv', index=False)
    
    print("Step 3 Complete: Labels applied based on clinical thresholds.")
    print(f"Iron Deficiency cases found: {df['iron_deficient'].sum()}")
    print(f"Vitamin D Deficiency cases found: {df['vit_d_deficient'].sum()}")

if __name__ == "__main__":
    apply_clinical_labels()