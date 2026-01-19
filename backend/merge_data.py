import pandas as pd

# 1. Define the file paths (Verify these match your downloaded filenames exactly)
DEMO_PATH = 'data/DEMO_J.XPT'    # Demographics
DIET_PATH = 'data/DR1TOT_J.XPT'  # Dietary Intake
FERRITIN_PATH = 'data/FERTIN_J.XPT' # Iron Biomarker
VITD_PATH = 'data/VID_J.XPT'     # Vitamin D Biomarker

def merge_nhanes_data():
    try:
        print("Reading NHANES files...")
        # Load Demographics: SEQN (ID), RIDAGEYR (Age), RIAGENDR (Gender)
        demo = pd.read_sas(DEMO_PATH)[['SEQN', 'RIDAGEYR', 'RIAGENDR']]
        
        # Load Dietary: SEQN, DR1TIRON (Iron mg), DR1TVD (Vit D mcg)
        diet = pd.read_sas(DIET_PATH)[['SEQN', 'DR1TIRON', 'DR1TVD']]
        
        # Load Labs: SEQN, LBDFERSI (Ferritin ug/L), LBXVIDMS (Vit D nmol/L)
        iron_lab = pd.read_sas(FERRITIN_PATH)[['SEQN', 'LBDFERSI']]
        vitd_lab = pd.read_sas(VITD_PATH)[['SEQN', 'LBXVIDMS']]

        print("Merging datasets...")
        # Start merging on SEQN
        merged = pd.merge(demo, diet, on='SEQN', how='inner')
        merged = pd.merge(merged, iron_lab, on='SEQN', how='inner')
        merged = pd.merge(merged, vitd_lab, on='SEQN', how='inner')

        # Drop rows with missing values to keep the data clean for the model
        merged = merged.dropna()

        # Save the result
        output_file = 'data/processed_nutrition.csv'
        merged.to_csv(output_file, index=False)
        
        print(f"Success! Master dataset saved to {output_file}")
        print(f"Total records processed: {len(merged)}")

    except Exception as e:
        print(f"Error: {e}. Ensure all .XPT files are in the 'data' folder.")

if __name__ == "__main__":
    merge_nhanes_data()