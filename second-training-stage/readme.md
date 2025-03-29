### Training

To train the models:

```bash
./runSecondStageTrain.sh
```


### Testing

1. **Full Model**:  
   Run the following to test the full model with all components:
   ```bash
   ./test_full_vf.sh
   ```

2. **Ablation - Without Data Augmentation**:  
   To test the model without data augmentation, use the following script:
   ```bash
   ./test_ablation_woa.sh
   ```

3. **Ablation - ResNet Architecture**:  
   To test a model with ResNet replacing the Transformer, run:
   ```bash
   ./test_ablation_restnet.sh
   ```

4. **Ablation - Without Value Encoder**:  
   For testing without the value encoder component, run:
   ```bash
   ./test_ablation_valem.sh
   ```
