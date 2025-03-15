### Training

To train the models, follow the steps below. Four different scripts are provided for training the full model and the ablation studies:

1. **Full Model**:  
   Run the following to train the full model with all components:
   ```bash
   python train.py
   ```

2. **Ablation - Without Data Augmentation**:  
   To train the model without data augmentation, use the following script:
   ```bash
   python ablationAugmentation.py
   ```

3. **Ablation - ResNet Architecture**:  
   To train a model with ResNet replacing the Transformer, run:
   ```bash
   python ablationResnet.py
   ```

4. **Ablation - Without Value Encoder**:  
   For training without the value encoder component, run:
   ```bash
   python ablationValEm.py
   ```

Ensure the appropriate datasets and configurations are set before running the scripts.
