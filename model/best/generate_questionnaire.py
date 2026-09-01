"""Generate synthetic questionnaire data correlated with Age for test2 demo"""
import pandas as pd, numpy as np, pathlib
from pathlib import Path
ROOT = Path("Vital-Age") if Path("Vital-Age/config.yaml").exists() else Path(".")
if not (ROOT/"config.yaml").exists():
    for p in [Path.cwd(), Path.cwd().parent]:
        if (p/"config.yaml").exists(): ROOT=p; break
df_raw = pd.read_csv(ROOT/"data/raw/blood_age_mega_raw.csv", usecols=["SEQN","Age"])
df_raw = df_raw.dropna(subset=["Age"])
df_raw = df_raw[(df_raw.Age>=18)&(df_raw.Age<=80)]
np.random.seed(42)
# generate correlated
n=len(df_raw)
gender = np.random.binomial(1, 0.49, n)  # 1=Male
height = 165 + gender*10 - df_raw.Age.values*0.06 + np.random.normal(0,6,n)
height = np.clip(height, 140, 200)
bmi = 23 + df_raw.Age.values*0.04 + np.random.normal(0,3,n)
bmi = np.clip(bmi, 15, 45)
weight = bmi * (height/100)**2 + np.random.normal(0,1.5,n)
waist = 75 + df_raw.Age.values*0.18 + (bmi-23)*1.2 + np.random.normal(0,5,n)
waist = np.clip(waist, 60, 140)
sbp = 105 + df_raw.Age.values*0.48 + gender*3 + np.random.normal(0,9,n)
sbp = np.clip(sbp, 90, 200)
# smoking: probs depend on age
smoking=[]
for a in df_raw.Age.values:
    p_never=0.55 - a*0.002
    p_former=0.20 + a*0.004
    p_current=1 - p_never - p_former
    p_current=max(0.05, p_current)
    p_never=max(0.2, p_never)
    # normalize
    s=sum([p_never,p_former,p_current]); p_never/=s; p_former/=s; p_current/=s
    smoking.append(np.random.choice(["Never","Former","Current"], p=[p_never,p_former,p_current]))
alcohol = np.random.randint(0,8,n)  # days/wk, weak negative with age
# add slight age correlation
alcohol = np.clip(alcohol - (df_raw.Age.values>60).astype(int),0,7)
exercise = np.clip(np.round(3.5 - df_raw.Age.values*0.025 + np.random.normal(0,1.5,n)),0,7).astype(int)

q=pd.DataFrame({"SEQN":df_raw.SEQN, "Gender":gender, "Weight":weight.round(1), "Height":height.round(1),
                "Waist":waist.round(1), "Systolic_BP":sbp.round(0).astype(int),
                "Smoking_status":smoking, "Alcohol_days":alcohol, "Exercise_days":exercise})
out=ROOT/"notebooks/blood/test2/questionnaire.csv"
q.to_csv(out, index=False)
print(f"Wrote {out} shape {q.shape}")
print(q.head().to_string())
print(q.describe(include='all').to_string())
