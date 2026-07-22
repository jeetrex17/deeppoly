Changes made to deeppoly_mnist.ipynb
====================================

1. **Model size increased**: 784→256→128→10 (was 784→64→32→10)
   - Wider layers give the model more capacity to learn large logit margins
   - Bigger clean margins survive DeepPoly over-approximation better

2. **Full training data**: TRAIN_LIMIT = 60000 (was 12000)
   - Uses all 60k MNIST training images instead of 20%

3. **More robust epochs**: ROBUST_EPOCHS = 30 (was 12)
   - Robust loss was still increasing at epoch 12, hadn't converged

4. **Higher robust loss weight**: ROBUST_LAMBDA = 1.0 (was 0.12)
   - Balances clean_loss and robust_loss equally so gradients push harder on certifying

5. **Better epsilon curriculum**: hits full ε by halfway through training
   - `current_eps = epsilon * min(1.0, epoch / (epochs // 2))`
   - Model spends half its training time at the actual evaluation epsilon

6. **Fixed masking value**: `-torch.inf` instead of `-1e9`
   - Proper -inf for logsumexp and max operations, no risk of numerical edge cases
