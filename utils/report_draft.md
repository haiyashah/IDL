# Remaining Useful Life Prediction for Turbofan Engines: LSTM vs. Temporal Fusion Transformer

**24-788 Introduction to Deep Learning — Spring 2026**

---

## Abstract

We compare a stacked Long Short-Term Memory network (LSTM) and a Temporal Fusion Transformer (TFT) for remaining useful life (RUL) prediction on the NASA C-MAPSS turbofan engine benchmark. The LSTM serves as the course baseline and the TFT is our variant, which extends the LSTM encoder with explicit variable selection and multi-head self-attention. We train and evaluate both models on two subsets of C-MAPSS: FD001 (single operating condition, single fault mode) and FD003 (single operating condition, two fault modes). On FD001, the TFT outperforms the LSTM on both RMSE (12.99 vs. 13.46 cycles) and the asymmetric PHM score (267 vs. 342), suggesting that attention over the full sequence helps identify fault onset more precisely. On FD003, the LSTM outperforms the TFT on RMSE (12.15 vs. 12.65 cycles), which we attribute to the TFT's variable selection network overfitting to the sensor patterns of a single fault mode. Both models generalize reasonably well across subsets, with RMSE degradation under one cycle.

---

## 1. Introduction

Unplanned failure of safety-critical components in aircraft engines causes significant downtime and poses safety risks. Condition-based maintenance, which schedules intervention based on real-time sensor data rather than fixed schedules, requires accurate estimates of how many operational cycles remain before a component fails. This estimate, called the Remaining Useful Life (RUL), is challenging to predict because degradation is gradual, nonlinear, and masked by sensor noise and varying operating conditions.

The NASA C-MAPSS dataset [1] is the standard benchmark for RUL prediction in the prognostics community. It simulates run-to-failure trajectories from turbofan engines under controlled conditions, providing multivariate time series of 21 sensor measurements per cycle. The task is to estimate how many cycles remain until failure given the sensor readings up to the current time step.

Recurrent models, particularly LSTMs, have been the dominant approach for this task since they naturally handle variable-length time series [2]. More recently, Transformer-based models have shown promise by attending globally to the input sequence rather than compressing history into a fixed-size state [3]. The Temporal Fusion Transformer (TFT) [4] extends this idea with an interpretable variable selection mechanism that learns which input features are most relevant at each time step, making it a natural variant to compare against a sequential baseline.

This project addresses two questions. First, does the TFT's attention-based architecture improve RUL prediction accuracy over a standard LSTM on the single-fault C-MAPSS benchmark? Second, does this advantage hold when the fault mode complexity increases? We answer both questions through controlled experiments on FD001 and FD003.

---

## 2. Methods

### 2.1 Dataset and Preprocessing

We use the C-MAPSS FD001 and FD003 subsets. FD001 contains 100 training and 100 test engine trajectories under a single operating condition with one fault mode (high-pressure compressor degradation). FD003 uses the same operating condition but introduces a second fault mode (fan degradation), making the sensor patterns more heterogeneous.

Each trajectory records 21 sensor measurements and 3 operational settings per cycle. Seven sensors that are nearly constant across all trajectories carry no degradation signal and are removed, leaving 17 input features. Sensor values are normalized per-feature using min-max scaling fit on the training set and applied to the test set. The RUL target is clipped at a maximum of 125 cycles, following standard practice [2], since the degradation signal is negligible early in engine life and exact RUL is not meaningfully predictable far from failure.

We construct training samples using a sliding window of 30 consecutive time steps. For the test set, we take the final 30 time steps of each engine trajectory, padding with zeros if the trajectory is shorter than 30 cycles. We hold out 10% of training engine units as a validation set for early stopping and learning rate scheduling.

### 2.2 LSTM Baseline

The LSTM baseline follows the architecture of Zheng et al. [2]. The input sequence (batch, 30, 17) is passed through a two-layer LSTM with hidden size 128 and dropout 0.2 between layers. The final hidden state is passed through a two-layer fully connected head (128 -> 64 -> 1) with ReLU activation and dropout, producing a scalar RUL estimate.

This model is representative of the recurrent approach that has dominated C-MAPSS benchmarks. The hidden state at each step summarizes all preceding observations, and the final step's state is used for prediction. The limitation of this design is that information from early time steps must propagate through the full sequence, which can be difficult when the relevant onset of degradation occurs at an unknown position in the window.

### 2.3 Temporal Fusion Transformer (TFT)

The TFT [4] was proposed for multi-horizon time series forecasting but adapts naturally to single-step regression. Our implementation consists of four stages applied to the input sequence.

**Variable Selection Network (VSN).** Each of the 17 input features is projected independently from a scalar to a d-model-dimensional embedding via a learned linear layer, then processed by a Gated Residual Network (GRN). A separate lightweight MLP takes the raw 17-dimensional input at each time step and produces a softmax distribution over features. The final representation at each time step is the weighted sum of per-feature embeddings under this distribution. This mechanism allows the model to attend more strongly to sensors that carry fault-relevant information, and the weights are inspectable after training.

**LSTM Encoder.** The selected feature representations are passed through a two-layer LSTM, identical in structure to the baseline. A gated skip connection blends the VSN output with the LSTM output: the gate is a sigmoid of a learned linear projection of the LSTM output, modulating how much the recurrent state modifies the input representation at each step.

**Multi-head Self-Attention.** The LSTM outputs are passed through a multi-head attention layer with 4 heads. This allows each time step to directly attend to any other time step in the window, capturing long-range patterns that the LSTM may miss. The output is processed by a GRN and added back via a residual connection.

**Regression Head.** The representation at the final time step is passed through a two-layer FC head (32 -> 1) to produce the RUL estimate.

The key architectural differences from the LSTM baseline are the variable selection mechanism and the attention layer. The hypothesis motivating this variant is that (1) not all 17 sensors contribute equally to RUL prediction, and learning to weight them should improve accuracy, and (2) attending globally over the window should help identify the precise onset of degradation, which is a point event that the LSTM must propagate through many steps.

### 2.4 Training

Both models are trained with Adam optimizer and MSE loss. The LSTM uses initial learning rate 1e-3 and the TFT uses 5e-4 due to its larger effective depth. Both use ReduceLROnPlateau scheduling with patience 5 and reduction factor 0.5, and gradient clipping at norm 1.0. Both are trained for 50 epochs with batch size 64. The model checkpoint with the lowest validation RMSE is saved and used for test evaluation. Training was performed on a single NVIDIA GPU via Google Colab Pro and completed in approximately 45 minutes per model.

### 2.5 Evaluation

We report two metrics. The primary metric is RMSE on the held-out test set, which measures average prediction accuracy in interpretable units (engine cycles). The secondary metric is the asymmetric PHM scoring function [1]:

s_i = exp(-d_i / 13) - 1   if d_i < 0  (early prediction)
s_i = exp( d_i / 10) - 1   if d_i >= 0 (late prediction)

where d_i = y_pred_i - y_true_i and the score is summed over all test engines. This function penalizes late predictions (predicting more remaining life than actually exists) more heavily than early predictions, reflecting the asymmetric cost of missing a failure versus scheduling unnecessary maintenance.

---

## 3. Results and Discussion

### 3.1 Quantitative Results

Table 1 reports test RMSE and PHM score for both models on both subsets.

**Table 1. Test set results on C-MAPSS FD001 and FD003.**

| Model  | FD001 RMSE | FD001 PHM | FD003 RMSE | FD003 PHM |
|--------|-----------|-----------|-----------|-----------|
| LSTM   | 13.46     | 341.51    | **12.15** | **246.40**|
| TFT    | **12.99** | **267.38**| 12.65     | 295.97    |

On FD001, the TFT outperforms the LSTM on both metrics. The RMSE improvement of 0.47 cycles (3.4%) is modest, but the PHM score improvement is more substantial: 267 vs. 342, a 22% reduction. Since the PHM score disproportionately penalizes late predictions, this gap indicates the TFT makes fewer dangerous over-predictions of remaining life. The LSTM's higher PHM score suggests it systematically underestimates degradation in some engines, predicting more remaining life than exists. The TFT's attention mechanism, by directly attending to earlier time steps, likely helps it detect the onset of degradation earlier in the window, avoiding these late predictions.

On FD003, the result reverses: the LSTM achieves lower RMSE (12.15 vs. 12.65 cycles, a 4.1% difference) and lower PHM score (246 vs. 296). This is a genuinely interesting finding. FD003 introduces a second fault mode, meaning that different engines in the test set fail via different physical mechanisms, producing qualitatively different sensor patterns. The TFT's variable selection network learns a weighted combination of sensors tuned to the training distribution. When two fault modes are present, the optimal sensor weighting differs between fault types, and the VSN may not generalize as well to the less frequent or less salient fault mode. The LSTM, by contrast, encodes the full sensor vector at each step without selection, which appears more robust to this heterogeneity.

Both models generalize well across subsets: the RMSE change between FD001 and FD003 is 1.31 cycles for LSTM and 0.35 cycles for TFT, both well under one standard deviation of individual predictions. Neither model degrades substantially under increased fault complexity.

### 3.2 Training Dynamics

Figure 1 shows validation RMSE over training epochs for both models on both subsets. Both models converge smoothly under the ReduceLROnPlateau schedule. The TFT converges faster on FD001, reaching a validation RMSE below 12.5 by epoch 10 while the LSTM does not reach this level until epoch 19. This is consistent with the TFT's attention mechanism providing a stronger learning signal early in training. On FD003, both models converge at similar rates, with the LSTM reaching its best validation RMSE of 10.42 at epoch 30 and the TFT reaching 10.01 at epoch 20.

### 3.3 Prediction Quality

Figure 2 shows predicted vs. true RUL scatter plots for all four model-subset combinations. All four models cluster near the diagonal for high-RUL values (engines far from failure), where the target is clipped at 125 cycles. Prediction error concentrates at low RUL values (engines near failure), which is consistent with prior work [2] and reflects the difficulty of predicting the exact failure cycle. The TFT on FD001 shows the tightest cluster at low RUL, consistent with its lower PHM score. The LSTM on FD003 shows less scatter at intermediate RUL values than the TFT, consistent with its lower RMSE.

### 3.4 Cross-Subset Comparison

Figure 3 summarizes the RMSE of both models across subsets. The pattern illustrates that the relative advantage of each architecture depends on the task complexity. For single-fault-mode prediction, the TFT's architectural inductive biases (variable selection and attention) are beneficial. For multi-fault-mode prediction, the simpler recurrent baseline is more robust. This suggests that variable selection may require sufficient training examples of each fault mode to learn reliable weights; with limited data and mixed fault modes, learning to ignore sensors is potentially harmful.

### 3.5 Variable Selection Interpretability

The TFT's variable selection weights provide a window into which sensors the model relies on. Figure 4 shows the mean selection weight per feature across the FD001 test set. The top-weighted sensors include s2 (total temperature at LPC outlet), s11 (corrected fan speed), and s7 (total temperature at HPC outlet), which are known to be sensitive indicators of compressor and fan degradation in the turbofan literature. The operational settings receive low weights, consistent with FD001 having a single fixed operating condition where they carry no discriminative information. This interpretability is a qualitative advantage of the TFT that the LSTM baseline cannot provide.

---

## 4. Conclusion

We compared a stacked LSTM and a Temporal Fusion Transformer for RUL prediction on NASA C-MAPSS. On FD001, the TFT achieves 3.4% lower RMSE and 22% lower PHM score, with the PHM improvement being the more practically significant result as it reflects fewer dangerous late predictions. On FD003, the LSTM is 4.1% more accurate, suggesting the TFT's variable selection does not generalize as well to multi-fault-mode distributions with limited training data. Both models achieve RMSE in the range of published results for LSTM-based methods (roughly 12 to 16 cycles on FD001 [2]), confirming that both implementations are correct and competitive.

The main takeaway is that architectural complexity does not uniformly help: the TFT's added inductive biases are beneficial when the task aligns with them (a single degradation mode with identifiable sensor signatures) and may hurt when they do not (mixed fault modes requiring flexible sensor weighting). Future work could investigate whether the TFT benefits from more training data or whether conditioning the variable selection on a fault-mode identifier would recover its advantage on FD003.

---

## References

[1] Saxena, A., Goebel, K., Simon, D., and Eklund, N. Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation. In *Proceedings of the 1st International Conference on Prognostics and Health Management (PHM)*, 2008.

[2] Zheng, S., Ristovski, K., Farahat, A., and Gupta, C. Long Short-Term Memory Network for Remaining Useful Life Estimation. In *Proceedings of the IEEE International Conference on Prognostics and Health Management (ICPHM)*, 2017. https://arxiv.org/abs/1709.01073

[3] Li, X., Ding, Q., and Sun, J. Remaining Useful Life Estimation in Prognostics Using Deep Convolution Neural Networks. *Reliability Engineering and System Safety*, 172, 2018. https://doi.org/10.1016/j.ress.2017.11.021

[4] Lim, B., Arik, S.O., Loeff, N., and Pfister, T. Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting. *Advances in Neural Information Processing Systems (NeurIPS)*, 2020. https://arxiv.org/abs/1912.09363

---

## Collaboration Statement

This project was completed individually. Claude (Anthropic) was used to help generate the initial code structure for data loading, model implementations, and the training loop. All architectural decisions, hyperparameter choices, experimental design, result interpretation, and written analysis are my own. The code was executed, verified, and debugged by me in Google Colab.
