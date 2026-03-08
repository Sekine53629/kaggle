# Kaggle Competition Rules Reference

This document summarizes standard Kaggle competition rules. Always verify the specific rules on each competition's "Rules" tab before starting.

## 1. External Data Policy

**Default**: External data is **NOT allowed** unless explicitly permitted.

### How to Check
- Go to the competition's **Rules** tab
- Look for "External Data" or "Additional Data" section
- If the competition has a **Discussion** thread titled "External Data", check it

### Common Patterns
| Competition Type | External Data | Notes |
|---|---|---|
| Featured (prize) | Usually NOT allowed | Strict rules for fairness |
| Research | Often allowed | Check rules carefully |
| Playground | Usually NOT allowed | Learning-focused |
| Getting Started | Varies | Check each competition |

### What Counts as External Data
- Datasets from other Kaggle competitions or datasets
- Web-scraped data
- Pre-trained model weights (e.g., ImageNet weights)
- API data (weather, census, etc.)
- Hand-labeled data

### What Does NOT Count as External Data
- Mathematical formulas and constants
- Domain knowledge encoded as features (e.g., `sin(month)` for seasonality)
- Standard libraries and their built-in datasets (e.g., stop words)
- Information derived solely from the provided data

## 2. Submission Rules

### Standard Limits
- **Daily submission limit**: Typically 5 per day (varies by competition)
- **Total submissions**: Usually unlimited
- **Final submissions**: Select up to 2 submissions for final evaluation
- **Team merges**: May reset submission count

### Important Notes
- Public leaderboard uses a subset of test data
- Private leaderboard (final) uses the remaining test data
- Overfitting to the public leaderboard is a common mistake

## 3. Team Rules

- **Maximum team size**: Varies (typically 3-5 for featured competitions)
- **Team merges**: Allowed before the merge deadline
- **One account per person**: Multiple accounts are strictly prohibited
- **Private sharing**: Cannot share code/data privately between teams

## 4. Code Requirements

### Code Competitions
- Some competitions require **notebook submissions** (code competitions)
- Must run within Kaggle's compute limits (GPU/CPU time)
- Internet access may be disabled during inference

### Standard Competitions
- Submit prediction files (CSV, etc.)
- Code is not required but encouraged for reproducibility

## 5. Prohibited Actions

- Using multiple accounts
- Sharing solutions privately between competing teams
- Exceeding computational limits in code competitions
- Using future data (data leakage from test set)
- Manual labeling of test data
- Reverse-engineering the test set

## 6. Intellectual Property

- Your code remains yours
- Competition data is licensed per competition terms
- Winners may be required to share their solution (for featured competitions)

## 7. Competition-Specific Checklist

Before starting any competition, verify:

- [ ] Read the **Rules** tab completely
- [ ] Check **External Data** policy
- [ ] Note the **submission limit** per day
- [ ] Check **team size** limit
- [ ] Verify if it's a **Code Competition** (notebook submission required)
- [ ] Check the **timeline** (merge deadline, final submission deadline)
- [ ] Read the **Evaluation** metric description carefully
- [ ] Check **Discussion** for rule clarifications from hosts

## 8. Per-Competition Notes

### recruit-restaurant-visitor-forecasting
- **Type**: Featured (ended)
- **External Data**: To be verified on competition page
- **Evaluation**: RMSLE
- **Note**: 11th place solution used weather data, suggesting external data may have been allowed. Verify before using.
- **Current approach**: Using only provided data + mathematical formulas (no external data)

---

**Source**: https://www.kaggle.com/docs/competitions
**Last updated**: 2026-03-08
**Note**: Always check the specific competition's Rules tab for the authoritative rules.
