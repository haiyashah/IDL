24-788 Mini Project Submission
================================
Haiya Niraj Shah | haiyas@andrew.cmu.edu

FILES IN THIS SUBMISSION
------------------------
CODE 24788_RUL_Project.ipynb   - Main notebook, run top to bottom in Colab \
report.tex      ( Inside Report_Latex_Overleaf.zip )          - LaTeX source for report (use with neurips_2023.sty) \
neurips_2023.sty      ( Inside Report_Latex_Overleaf.zip )  - Style file for LaTeX \
cmapss_project.zip        - All model code (upload to Colab when notebook asks) 

HOW TO RUN
----------
1. Open CODE 24788_RUL_Project.ipynb in Google Colab
2. Set runtime to GPU (Runtime > Change runtime type > T4 GPU)
3. Run cells top to bottom
4. When cell asks to upload cmapss_project.zip, upload it
5. Training takes about 60-80 minutes total (4 model/subset combinations)
6. All figures save to cmapss_project/figures/

TO REPRODUCE FIGURES WITHOUT RETRAINING
----------------------------------------
If you have the checkpoints and result JSONs already:
1. Upload cmapss_project.zip AND your checkpoints/results folders to Colab
2. Skip the training cell (Step 7)
3. Run from Step 8 onwards

WHAT THE CODE DOES
------------------
- Trains LSTM (baseline) and TFT (variant) on NASA C-MAPSS FD001 and FD003
- Evaluates using RMSE and asymmetric PHM score
- Generates 6 figures: training curves, scatter plots, cross-subset comparison,
  variable importance, error analysis, error distribution
- Saves all results to JSON files for reproducibility

DEPENDENCIES
------------
torch, numpy, pandas, scikit-learn, matplotlib, wandb
All available in Google Colab (wandb installed in first cell)
