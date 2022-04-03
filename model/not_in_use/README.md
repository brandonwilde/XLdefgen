## Train model WITH Trainer API

Model not currently in use. If using, you may need to update the  
dataset path or run the script from the model directory.

Example usage:

```bash
python run_translation_with_trainer.py trainer_test_args.json
```

OR (for testing on small GPU)

```bash
python run_translation_with_trainer.py --file train_smalltest_args.json
```

Can use HF 'stas/mt5-tiny-random' model for testing.